"""Pitch detector: combines NanoPitchEncoder backbone and PitchHead with Viterbi decoding."""

import numpy as np
from typing import Any

from ml_new.legacy_ml.feature_extraction.audio_utils import load_audio
from ml_new.legacy_ml.pitch_detection.model import PitchHead


# ── Viterbi decoders ─────────────────────────────────────────────────────────

def viterbi_decode(posteriorgram, transition_width=12, voicing_threshold=0.3,
                   onset_penalty=2.0, vad=None, vad_weight=0.0):
    """Decode a pitch posteriorgram into a smooth f0 track using Viterbi.

    The Viterbi algorithm finds the most likely sequence of pitch states
    over time, given:
      - Observation probabilities: the model's pitch posteriorgram
      - Transition constraints: pitch can't jump more than ±12 bins per frame
      - Voicing model: an unvoiced state with onset/offset penalties

    This is a proper dynamic programming algorithm (not just argmax), so
    it produces smoother pitch tracks and handles brief dropouts better.

    State space: 360 voiced bins + 1 unvoiced state = 361 total.
    Transition: voiced states can reach neighbors within ±transition_width.
    Observation: log(posteriorgram) for voiced, log(1 - max_post) for unvoiced.

    Vectorized implementation using numpy strided windows for speed.

    Args:
        posteriorgram: (T, 360) pitch probabilities from the model
        transition_width: max pitch change per frame in bins (12 = 240 cents)
        voicing_threshold: min confidence to initialize as voiced
        onset_penalty: log-domain cost for voiced↔unvoiced transitions
        vad: optional (T,) VAD probability array for guidance
        vad_weight: weight for blending VAD into voicing observation

    Returns:
        f0_hz: (T,) float32 array of decoded f0 in Hz (0 = unvoiced)
    """
    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1  # window size for transition neighbourhood
    log_obs = np.log(posteriorgram + 1e-10)

    V = np.full((T, N + 1), -np.inf, dtype=np.float64)
    bp = np.zeros((T, N + 1), dtype=np.int32)

    max_post = posteriorgram[0].max()
    if max_post > voicing_threshold:
        V[0, :N] = log_obs[0]
    V[0, N] = np.log(1.0 - max_post + 1e-10)

    for t in range(1, T):
        max_post_t = posteriorgram[t].max()
        prev = V[t - 1, :N]

        padded = np.pad(prev, (tw, tw), constant_values=-np.inf)
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        best_k = np.argmax(windows, axis=1)
        best_val = windows[np.arange(N), best_k]
        best_from_voiced = np.clip(np.arange(N) - tw + best_k, 0, N - 1)

        from_unvoiced = V[t - 1, N] - onset_penalty

        use_voiced = best_val >= from_unvoiced
        V[t, :N] = np.where(use_voiced, best_val, from_unvoiced) + log_obs[t]
        bp[t, :N] = np.where(use_voiced, best_from_voiced, N)

        best_voiced_score = prev.max()
        best_voiced_idx = prev.argmax()
        from_voiced = best_voiced_score - onset_penalty
        stay_uv = V[t - 1, N]
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        if stay_uv >= from_voiced:
            V[t, N] = stay_uv + uv_obs
            bp[t, N] = N
        else:
            V[t, N] = from_voiced + uv_obs
            bp[t, N] = best_voiced_idx

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    f0_hz = np.zeros(T, dtype=np.float32)
    voiced_mask = path < N
    if voiced_mask.any():
        f0_hz[voiced_mask] = _bin_to_f0(path[voiced_mask].astype(np.float64))

    return f0_hz


def viterbi_decode_realtime(posteriorgram, transition_width=12,
                            voicing_threshold=0.3, onset_penalty=2.0):
    """Realtime (greedy) Viterbi — matches the C/WASM deployment exactly.

    Unlike the offline version, this processes frames left-to-right and
    emits the best state immediately at each frame, without backtracing.
    This is what runs in real-time in the browser.

    The tradeoff: realtime Viterbi can't "change its mind" about earlier
    frames when it sees future evidence. In practice, the difference is
    small for well-trained models, but it can occasionally miss brief
    voiced segments or produce slightly less smooth pitch tracks.

    Same transition model as offline:
      - 360 voiced bins + 1 unvoiced state
      - ±transition_width bin transitions allowed per frame
      - onset/offset penalty for voiced↔unvoiced switches

    Args / Returns: same as viterbi_decode.
    """
    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1

    prev = np.full(N + 1, -np.inf, dtype=np.float64)
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_post_t = posteriorgram[t].max()
        log_obs_t = np.log(posteriorgram[t] + 1e-10)
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        curr = np.full(N + 1, -np.inf, dtype=np.float64)

        if t == 0:
            if max_post_t > voicing_threshold:
                curr[:N] = log_obs_t
            curr[N] = uv_obs
        else:
            prev_voiced = prev[:N]

            padded = np.pad(prev_voiced, (tw, tw), constant_values=-np.inf)
            windows = np.lib.stride_tricks.as_strided(
                padded, shape=(N, W),
                strides=(padded.strides[0], padded.strides[0]))
            best_val = np.max(windows, axis=1)

            from_uv = prev[N] - onset_penalty

            curr[:N] = np.maximum(best_val, from_uv) + log_obs_t

            from_voiced = prev_voiced.max() - onset_penalty
            curr[N] = max(prev[N], from_voiced) + uv_obs

        best_state = np.argmax(curr)
        if best_state < N:
            f0_hz[t] = _bin_to_f0(float(best_state))

        prev = curr

    return f0_hz


# ── Pitch conversion helpers ──────────────────────────────────────────────────


def _bin_to_f0(bins: np.ndarray | float) -> np.ndarray | float:
    """Convert NanoPitch bin index to Hz. Inverse of f0_to_bin."""
    return PitchHead.FMIN * 2.0 ** (np.asarray(bins) * PitchHead.CENTS_PER_BIN / 1200.0)


# ── Detector ─────────────────────────────────────────────────────────────────

class PitchDetector:
    """Extracts pitch features from audio using NanoPitchEncoder + PitchHead + Viterbi.

    When a trained ``VoiceCoachModel`` (or ``PitchHead``) is provided, the model
    path produces per-frame pitch via Viterbi decoding.  Falls back to
    ``librosa.yin`` when no model is available.

    Args:
        model: Optional ``VoiceCoachModel`` instance providing pitch predictions.
        sr: Sample rate the detector operates at.
    """

    def __init__(self, model=None, sr: int = 16000) -> None:
        self._model = model
        self._sr = sr
        if model is not None:
            from ml_new.legacy_ml.feature_extraction.mel import MelExtractor
            self._mel_extractor = MelExtractor(sr=sr)

    def analyze(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict[str, Any]:
        """Extract pitch features from a single audio chunk.

        Args:
            audio_chunk: 1-D float32 numpy array of audio samples.
            sr: Sample rate of the audio chunk.

        Returns:
            Dict with keys: stability_score, estimated_notes, pitch_curve,
            note_transitions.
        """
        if self._model is not None:
            return self._analyze_with_model(audio_chunk, sr)
        return self._analyze_with_librosa(audio_chunk, sr)

    def _analyze_with_model(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Run model forward pass and apply realtime Viterbi decoding."""
        import torch

        mel_np = self._mel_extractor.compute(audio)  # (n_mels, T)
        mel_t = torch.from_numpy(mel_np).T.unsqueeze(0)  # (1, T, 40)

        self._model.eval()
        with torch.no_grad():
            out = self._model(mel_t)

        pitch_logits = out["pitch_logits"]  # (1, T, 360)
        probs = torch.sigmoid(pitch_logits).squeeze(0).cpu().numpy()  # (T, 360)

        f0_hz = viterbi_decode_realtime(probs)

        return self._build_output(f0_hz)

    def _analyze_with_librosa(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        import librosa

        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        return self._build_output(f0)

    @staticmethod
    def _build_output(f0: np.ndarray) -> dict[str, Any]:
        import librosa

        voiced = f0[f0 > 0]
        if len(voiced) > 1:
            stability = float(np.clip(1.0 - np.std(voiced) / (np.mean(voiced) + 1e-8), 0.0, 1.0))
        else:
            stability = 0.0

        estimated_notes: list[str] = []
        note_transitions: list[tuple[str, str]] = []
        if len(voiced) > 0:
            try:
                notes = [librosa.hz_to_note(float(hz)) for hz in voiced]
                estimated_notes = list(dict.fromkeys(notes))
                note_transitions = [(a, b) for a, b in zip(notes[:-1], notes[1:]) if a != b]
            except Exception:
                pass

        return {
            "stability_score": stability,
            "estimated_notes": estimated_notes,
            "pitch_curve": f0.tolist(),
            "note_transitions": note_transitions,
        }


def extract_pitch_features(audio_path: str) -> dict[str, Any]:
    """Load audio and extract pitch features. Module-level entry point for pipeline.py.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with keys: stability_score, estimated_notes, pitch_curve, note_transitions.
    """
    audio = load_audio(audio_path, sr=16000)
    detector = PitchDetector()
    return detector.analyze(audio, sr=16000)
