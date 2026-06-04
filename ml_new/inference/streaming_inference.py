"""Stateful streaming inference for UnifiedVocalModel.

Internally batches 20 frames (200 ms) per HCQT extraction call — the CQT
has a ~150 ms fixed overhead so single-frame extraction is 15x slower than
real-time; at 20-frame batches it runs at ~1.3x real-time (enough headroom).

Public API: push one 160-sample PCM frame at a time.  Results are buffered
internally; the engine returns None for 19 out of 20 calls and a list of 20
LiveFrames on the 20th.  The WebSocket handler should send all 20 in order.

Usage::

    eng = StreamingEngine(checkpoint_path="ml_new/checkpoints/unified_v2/best.pt")
    eng.reset()

    for pcm_frame in audio_chunks:           # float32 (160,) @ 16kHz
        batch = eng.push_frame(pcm_frame)    # None most of the time
        if batch:
            for frame in batch:
                send_to_client(frame)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.feature_extraction.hcqt import HCQTExtractor
from ml_new.feature_extraction.vad_features import VADFeatureExtractor
from ml_new.models.unified_model import UnifiedVocalModel, TECHNIQUE_VOCAB
from ml_new.inference.coach_inference import VOICED_THRESH, BREATH_THRESH, ONSET_THRESH

SR         = 16_000
HOP        = 160        # 10 ms per frame
N_BINS     = 180
BPO        = 36
FMIN       = UnifiedVocalModel.FMIN   # 32.7 Hz

# Batch size: process this many frames at once for efficient HCQT extraction.
# CQT has ~150ms fixed overhead; 20 frames (200ms audio) gives ~1.4x realtime.
# Total latency to user: ~200ms buffering + ~144ms processing ≈ 344ms.
BATCH_FRAMES = 20
BATCH_SAMPLES = BATCH_FRAMES * HOP   # 3200 samples = 200ms

# Past audio context prepended to each batch so the CQT edge frames are valid.
# One extra batch-length of context is sufficient.
CONTEXT_SAMPLES = BATCH_SAMPLES

# Vibrato: measure over a 600 ms window, looking for oscillation at 4–8 Hz
VIBRATO_WINDOW_FRAMES = 60      # 600 ms
VIBRATO_RATE_MIN_HZ   = 4.0
VIBRATO_RATE_MAX_HZ   = 8.0

# Tempo: keep the last 16 inter-onset intervals
ONSET_IBI_QUEUE = 16

# Onset threshold for streaming (tuned for precision/recall balance)
ONSET_THRESH_STREAM = 0.50


class LiveFrame(TypedDict):
    t_ms:               float   # timestamp in ms from session start
    pitch_hz:           float   # 0.0 = unvoiced
    voiced:             bool
    loudness_db:        float   # RMS in dBFS; −60 = silence
    breath:             bool
    onset:              bool
    vibrato_rate_hz:    float   # 0.0 = no vibrato detected
    vibrato_depth_cents: float
    tempo_bpm:          float   # 0.0 = not enough onsets yet
    technique:          str     # best-guess technique (updates every frame)
    technique_conf:     float
    proc_ms:            float   # wall-clock ms to process this batch (RTF = proc_ms / (BATCH_FRAMES * 10))


class StreamingEngine:
    """Stateful streaming inference engine.

    Maintains all per-session state. Call reset() at the start of each
    recording; then push_frame() for every 160-sample audio chunk.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        dev_str = device
        if dev_str == "auto":
            if torch.backends.mps.is_available():
                dev_str = "mps"
            elif torch.cuda.is_available():
                dev_str = "cuda"
            else:
                dev_str = "cpu"
        self.device = torch.device(dev_str)

        self.model = UnifiedVocalModel().to(self.device)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.eval()

        self.bin_hz = self.model.bin_hz.cpu().numpy()

        # Feature extractors — stateless, reused across frames
        self._hcqt_ext = HCQTExtractor(
            sr=SR, hop_length=HOP, n_bins=N_BINS, bins_per_octave=BPO
        )
        self._vad_ext = VADFeatureExtractor(sr=SR, hop_length=HOP)

        # Session state (reset per recording)
        self._h: torch.Tensor | None = None
        # Past context prepended to each batch so CQT edge frames are valid
        self._ctx_buf: np.ndarray = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        # Accumulate incoming frames until we have a full batch
        self._pending: list[np.ndarray] = []
        self._frame_count: int = 0
        self._f0_window: deque[float] = deque(maxlen=VIBRATO_WINDOW_FRAMES)
        self._onset_times_ms: deque[float] = deque(maxlen=ONSET_IBI_QUEUE)
        self._prev_onset: bool = False
        self._tech_accumulator: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all session state. Call at the start of each recording."""
        self._h = self.model.init_hidden(batch_size=1, device=self.device)
        self._ctx_buf = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._pending = []
        self._frame_count = 0
        self._f0_window.clear()
        self._onset_times_ms.clear()
        self._prev_onset = False
        self._tech_accumulator = None

    def push_frame(self, pcm: np.ndarray) -> list[LiveFrame] | None:
        """Accept one 160-sample frame; return a batch of LiveFrames every 20 frames.

        Returns None for the first 19 calls in each batch, and a list of 20
        LiveFrames on the 20th.  The caller should send all frames in order.

        Args:
            pcm: float32 array of HOP (160) samples at 16kHz.
        """
        pcm = np.asarray(pcm, dtype=np.float32)
        if len(pcm) < HOP:
            pcm = np.pad(pcm, (0, HOP - len(pcm)))
        pcm = pcm[:HOP]

        self._pending.append(pcm)
        self._frame_count += 1

        if len(self._pending) < BATCH_FRAMES:
            return None

        return self._process_batch()

    def flush(self) -> list[LiveFrame] | None:
        """Process any remaining buffered frames at end of recording.

        Returns a partial batch (< BATCH_FRAMES) or None if nothing pending.
        """
        if not self._pending:
            return None
        # Pad to a full batch with silence so the CQT always sees enough audio
        while len(self._pending) < BATCH_FRAMES:
            self._pending.append(np.zeros(HOP, dtype=np.float32))
        return self._process_batch()

    def _process_batch(self) -> list[LiveFrame]:
        """Process the current pending batch and return per-frame results."""
        t_batch_start = time.perf_counter()
        batch_audio = np.concatenate(self._pending)         # (BATCH_SAMPLES,)
        self._pending = []

        # Prepend context so CQT edge frames are valid
        full_audio = np.concatenate([self._ctx_buf, batch_audio])  # (CONTEXT+BATCH,)

        # Update context buffer for next batch
        self._ctx_buf = full_audio[-CONTEXT_SAMPLES:]

        # Extract features for the full window; keep only the BATCH_FRAMES new frames
        hcqt = self._hcqt_ext.compute(full_audio)           # (6, 180, T_total)
        vad  = self._vad_ext.compute(full_audio)             # (3, T_total)
        T = min(hcqt.shape[2], vad.shape[1])

        # The last BATCH_FRAMES columns correspond to the new frames
        hcqt_new = hcqt[:, :, max(0, T - BATCH_FRAMES): T]
        vad_new  = vad[:,    max(0, T - BATCH_FRAMES): T]
        n_new    = hcqt_new.shape[2]

        hcqt_t = torch.from_numpy(hcqt_new).unsqueeze(0).to(self.device)  # (1,6,180,n)
        vad_t  = torch.from_numpy(vad_new).unsqueeze(0).to(self.device)   # (1,3,n)

        with torch.no_grad():
            pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, self._h = \
                self.model(hcqt_t, vad_t, self._h)

        # Technique: exponential moving average for temporal stability
        raw_logits = tech_logits[0].cpu().numpy()           # (n_techniques,)
        alpha = 0.05
        if self._tech_accumulator is None:
            self._tech_accumulator = raw_logits.copy()
        else:
            self._tech_accumulator = (1 - alpha) * self._tech_accumulator + alpha * raw_logits
        tech_probs = np.exp(self._tech_accumulator - self._tech_accumulator.max())
        tech_probs /= tech_probs.sum()
        tech_idx = int(np.argmax(tech_probs))
        tech_name = TECHNIQUE_VOCAB[tech_idx]
        tech_conf = float(tech_probs[tech_idx])

        proc_ms = (time.perf_counter() - t_batch_start) * 1000.0

        # Build per-frame results
        base_t = (self._frame_count - BATCH_FRAMES) * 10.0
        results: list[LiveFrame] = []

        for i in range(n_new):
            t_ms     = base_t + i * 10.0
            pcm_i    = batch_audio[i * HOP: (i + 1) * HOP]
            pitch_bin = int(pitch_logits[0, i].argmax().item())
            pitch_raw = float(self.bin_hz[pitch_bin])
            v_prob   = float(voiced_prob[0, i].item())
            b_prob   = float(breath_prob[0, i].item())
            o_prob   = float(onset_prob[0, i].item())

            voiced   = v_prob >= VOICED_THRESH
            pitch_hz = pitch_raw if voiced else 0.0
            breath   = b_prob >= BREATH_THRESH
            onset    = (o_prob >= ONSET_THRESH_STREAM) and not self._prev_onset

            rms = float(np.sqrt(np.mean(pcm_i ** 2)) + 1e-9)
            loudness_db = float(20.0 * np.log10(rms))

            self._f0_window.append(pitch_hz if voiced else 0.0)
            vibrato_rate, vibrato_depth = self._compute_vibrato()

            if onset:
                self._onset_times_ms.append(t_ms)
            tempo_bpm = self._compute_tempo()

            self._prev_onset = o_prob >= ONSET_THRESH_STREAM

            results.append(LiveFrame(
                t_ms=t_ms,
                pitch_hz=pitch_hz,
                voiced=voiced,
                loudness_db=loudness_db,
                breath=breath,
                onset=onset,
                vibrato_rate_hz=vibrato_rate,
                vibrato_depth_cents=vibrato_depth,
                tempo_bpm=tempo_bpm,
                technique=tech_name,
                technique_conf=tech_conf,
                proc_ms=proc_ms,
            ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _silent_frame(self, t_ms: float) -> LiveFrame:
        return LiveFrame(
            t_ms=t_ms, pitch_hz=0.0, voiced=False, loudness_db=-60.0,
            breath=False, onset=False, vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0, tempo_bpm=0.0,
            technique="unknown", technique_conf=0.0, proc_ms=0.0,
        )

    def _compute_vibrato(self) -> tuple[float, float]:
        """Autocorrelation-based vibrato rate and depth over the F0 window."""
        f0_arr = np.array(self._f0_window, dtype=np.float64)
        voiced = f0_arr > 0
        if voiced.sum() < VIBRATO_WINDOW_FRAMES // 2:
            return 0.0, 0.0

        # Cents deviation from median
        median_f0 = np.median(f0_arr[voiced])
        if median_f0 <= 0:
            return 0.0, 0.0
        cents = np.where(voiced, 1200.0 * np.log2(np.clip(f0_arr, 1e-6, None) / median_f0), 0.0)

        # Zero-mean autocorrelation
        cents -= cents.mean()
        n = len(cents)
        corr = np.correlate(cents, cents, mode="full")[n - 1:]
        if corr[0] == 0:
            return 0.0, 0.0
        corr /= corr[0]

        # Look for peaks in the 4–8 Hz range (12–25 frames at 10ms hop)
        min_lag = int(SR / (HOP * VIBRATO_RATE_MAX_HZ))  # ~12 frames
        max_lag = int(SR / (HOP * VIBRATO_RATE_MIN_HZ))  # ~25 frames

        if max_lag >= len(corr):
            return 0.0, 0.0

        search = corr[min_lag: max_lag + 1]
        peak_offset = int(np.argmax(search))
        peak_val = float(search[peak_offset])

        if peak_val < 0.3:   # weak correlation → not vibrato
            return 0.0, 0.0

        lag = min_lag + peak_offset
        rate = float(SR / (HOP * lag))
        depth = float(np.std(cents[voiced]) * 2.0)  # peak-to-peak ≈ 2σ
        return round(rate, 2), round(depth, 1)

    def _compute_tempo(self) -> float:
        """Estimate BPM from the most recent inter-onset intervals."""
        times = list(self._onset_times_ms)
        if len(times) < 4:
            return 0.0
        ibi_ms = np.diff(times)
        # Ignore extreme IBIs (< 200ms = >300 BPM, > 2000ms = <30 BPM)
        ibi_ms = ibi_ms[(ibi_ms >= 200) & (ibi_ms <= 2000)]
        if len(ibi_ms) < 3:
            return 0.0
        return round(60_000.0 / float(np.median(ibi_ms)), 1)
