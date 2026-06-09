"""Coaching inference module for VocalStars.

Takes a raw audio file and returns a CoachingResult with:
  - Per-frame pitch, voicing, breath and onset arrays (from the neural model)
  - Clip-level technique classification
  - Note-level pitch analysis  (signal processing — segment_notes)
  - Voice quality metrics      (PRAAT via parselmouth — HNR, jitter, shimmer)
  - Vibrato detection          (F0 autocorrelation — per sustained note)
  - Human-readable score, summary, issues and exercises

Usage
-----
    from ml_new.inference.coach_inference import analyse_recording
    result = analyse_recording("my_singing.wav",
                               checkpoint="ml_new/checkpoints/unified/best.pt")
    print(result.summary)

CLI
---
    python -m ml_new.inference.coach_inference recording.wav
    python -m ml_new.inference.coach_inference recording.wav \\
        --checkpoint ml_new/checkpoints/unified/best.pt --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.feature_extraction.hcqt import HCQTExtractor
from ml_new.feature_extraction.vad_features import VADFeatureExtractor
from ml_new.feature_extraction.breath_labels import derive_breath_labels
from ml_new.feature_extraction.onset_labels import derive_onset_labels
from ml_new.models.unified_model import UnifiedVocalModel, TECHNIQUE_VOCAB
from ml_new.models.acoustic_technique import (
    AcousticTechniqueClassifier, extract_acoustic_features,
)
from ml_new.inference.algorithms import (
    NoteSegment, VoiceQuality, VibratoInfo,
    segment_notes_for_coaching, extract_voice_quality,
    flat_notes_summary, vibrato_summary,
)

SR         = 16_000
HOP_LENGTH = 160
HOP_S: float = HOP_LENGTH / SR   # 0.01 s per frame

# ---------------------------------------------------------------------------
# Robust audio loader (handles WAV, WebM/Opus, MP4, OGG, etc.)
# ---------------------------------------------------------------------------

def _load_audio_robust(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load audio from any container/codec using soundfile first, then PyAV.

    librosa's audioread fallback requires ffmpeg which is not always installed.
    soundfile handles WAV/FLAC/OGG-Vorbis; PyAV handles WebM/Opus and MP4/AAC.
    """
    import soundfile as sf

    path = str(path)
    try:
        data, native_sr = sf.read(path, always_2d=False, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        data, native_sr = _load_audio_pyav(path)

    if native_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=native_sr, target_sr=sr)

    return np.asarray(data, dtype=np.float32)


def _load_audio_pyav(path: str) -> tuple[np.ndarray, int]:
    """Decode audio via PyAV (bundles its own libav codecs — no system ffmpeg needed)."""
    import av as _av  # lazy import; not a hard dependency for the model path

    frames: list[np.ndarray] = []
    native_sr = 48000
    with _av.open(path) as container:
        stream = container.streams.audio[0]
        native_sr = stream.rate or 48000
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            # Float planar (fltp) shape: (channels, samples)
            # Integer formats shape: (channels, samples) or (samples,)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            # Normalize integer PCM to float32 [-1, 1]
            if arr.dtype.kind == "i":
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            elif arr.dtype.kind == "u":
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max) * 2.0 - 1.0
            else:
                arr = arr.astype(np.float32)
            frames.append(arr)
    audio = np.concatenate(frames) if frames else np.zeros(native_sr, dtype=np.float32)
    return audio, native_sr
N_BINS        = 180
BINS_PER_OCTAVE = 36
FMIN = UnifiedVocalModel.FMIN    # 32.7 Hz (C1)

VOICED_THRESH = 0.50
BREATH_THRESH = 0.35
ONSET_THRESH  = 0.30

MIN_PHRASE_S  = 0.5   # phrases shorter than this are ignored
MIN_SILENCE_S = 0.10  # unvoiced gap needed to split phrases


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

TASK_TYPES = {
    "free_singing",
    "reference_song",
    "sustained_note",
    "pitch_slide",
    "scale",
    "interval",
    "rhythm",
    "breath_control",
    "tone_consistency",
}


@dataclass
class TaskConfig:
    task_type: str = "free_singing"
    skill_focus: str | list[str] | None = None
    target: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    scoring_mode: str = "auto"
    strictness: float | str | None = None

@dataclass
class CoachingResult:
    """Full coaching analysis for one recorded clip."""

    # ── Raw per-frame arrays (10 ms / frame) ──────────────────────────────
    pitch_hz:      np.ndarray  # float32, 0.0 = unvoiced
    voiced:        np.ndarray  # bool
    breath_frames: np.ndarray  # bool
    onset_frames:  np.ndarray  # bool
    hop_s:         float       # seconds per frame (always 0.01)

    # ── Clip-level technique (neural model) ───────────────────────────────
    technique:            str
    technique_confidence: float
    all_technique_scores: dict[str, float]

    # ── Derived coaching metrics ──────────────────────────────────────────
    pitch_accuracy:    float        # fraction of voiced frames within 50 ¢ of nearest semitone
    pitch_drift_cents: float        # median signed cents offset (+ = sharp, − = flat)
    phrase_lengths_s:  list[float]  # seconds per phrase
    breath_count:      int
    onset_count:       int
    onset_clarity:     float        # mean onset probability at detected onset peaks

    # ── Algorithmic enrichments ───────────────────────────────────────────
    notes:         list[NoteSegment]   # individual note events with per-note pitch info
    voice_quality: VoiceQuality | None # HNR, jitter, shimmer (None if parselmouth unavailable)
    vibrato_stats: dict                # from vibrato_summary()
    diagnostics:   dict                # summary-only model/f0/postprocessing diagnostics
    analysis_validity: dict            # postprocessing gate for score/coaching safety
    task_config: dict
    task_analysis: dict

    # ── Human-readable coaching output ───────────────────────────────────
    score:     int | None  # legacy/display score; see full_song_score/diagnostic_score
    full_song_score: int | None
    diagnostic_score: int | None
    score_status: str
    score_caveat: str | None
    summary:   str         # one-sentence overview
    issues:    list[str]   # up to 4 specific problems detected
    exercises: list[str]   # one targeted exercise per issue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_recording(
    audio_path: str | Path,
    checkpoint: str | Path | None = None,
    device:     str = "cpu",
    task_config: TaskConfig | dict[str, Any] | None = None,
) -> CoachingResult:
    """Analyse a singing recording and return coaching feedback.

    Args:
        audio_path: Path to any audio file supported by librosa (.wav, .mp3, …).
        checkpoint: Path to a trained ``UnifiedVocalModel`` ``.pt`` checkpoint.
            Falls back to librosa.pyin + heuristics when ``None`` or missing.
        device: PyTorch device string (``"cpu"``, ``"mps"``, ``"cuda"``).

    Returns:
        :class:`CoachingResult` with raw arrays, algorithmic metrics, and
        human-readable coaching advice.
    """
    audio = _load_audio_robust(audio_path, sr=SR)

    # Guard against clips that are shorter than the model's minimum context
    # window (1024-sample pYIN frame). Pad to at least 1 second with silence
    # so all downstream extractors receive a valid-length signal.
    MIN_SAMPLES = SR  # 1 second at 16 kHz
    if len(audio) < MIN_SAMPLES:
        audio = np.pad(audio, (0, MIN_SAMPLES - len(audio)), mode="constant")

    ckpt_path = Path(checkpoint) if checkpoint else None
    use_model = (ckpt_path is not None) and ckpt_path.exists()

    if use_model:
        result = _run_model(audio, ckpt_path, device, task_config=task_config)
    else:
        result = _run_fallback(audio, task_config=task_config)
    return result


# ---------------------------------------------------------------------------
# Model inference path
# ---------------------------------------------------------------------------

def _load_acoustic_classifier(
    ckpt_path: Path, device: torch.device
) -> AcousticTechniqueClassifier | None:
    """Load the acoustic technique classifier if its checkpoint exists alongside the backbone."""
    acoustic_ckpt = ckpt_path.parent / "acoustic_best.pt"
    if not acoustic_ckpt.exists():
        # Also check the unified_tech directory
        acoustic_ckpt = ckpt_path.parent.parent / "unified_tech" / "acoustic_best.pt"
    if not acoustic_ckpt.exists():
        return None
    clf = AcousticTechniqueClassifier().to(device)
    state = torch.load(str(acoustic_ckpt), map_location=device, weights_only=True)
    clf.load_state_dict(state.get("classifier_state_dict", state))
    clf.eval()
    return clf


def _run_model(
    audio: np.ndarray,
    ckpt_path: Path,
    device: str,
    task_config: TaskConfig | dict[str, Any] | None = None,
) -> CoachingResult:
    hcqt_ext = HCQTExtractor(sr=SR, hop_length=HOP_LENGTH,
                              n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    vad_ext  = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)

    hcqt      = hcqt_ext.compute(audio)
    vad_feats = vad_ext.compute(audio)
    T = min(hcqt.shape[2], vad_feats.shape[1])
    hcqt      = hcqt[:, :, :T]
    vad_feats = vad_feats[:, :T]

    dev   = torch.device(device)
    model = UnifiedVocalModel().to(dev)
    ckpt  = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    hcqt_t  = torch.from_numpy(hcqt).unsqueeze(0).to(dev)
    vad_t   = torch.from_numpy(vad_feats).unsqueeze(0).to(dev)

    with torch.no_grad():
        pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits_base, _ = model(
            hcqt_t, vad_t,
        )
        pitch_probs = torch.softmax(pitch_logits[0], dim=-1)
        pitch_top2 = torch.topk(pitch_probs, k=2, dim=-1).values
        pitch_entropy = -(pitch_probs * torch.log(pitch_probs.clamp_min(1e-12))).sum(dim=-1)

    bin_hz_np  = model.bin_hz.cpu().numpy()
    pitch_hz   = bin_hz_np[pitch_logits[0].argmax(dim=-1).cpu().numpy()].astype(np.float32)
    voiced_np  = voiced_prob[0].cpu().numpy()
    breath_np  = breath_prob[0].cpu().numpy()
    onset_np   = onset_prob[0].cpu().numpy()
    pitch_conf_np = pitch_top2[:, 0].cpu().numpy()
    pitch_margin_np = (pitch_top2[:, 0] - pitch_top2[:, 1]).cpu().numpy()
    pitch_entropy_np = pitch_entropy.cpu().numpy()

    voiced_bool = voiced_np >= VOICED_THRESH
    pitch_hz    = np.where(voiced_bool, pitch_hz, 0.0).astype(np.float32)
    breath_bool = breath_np >= BREATH_THRESH
    onset_bool  = onset_np  >= ONSET_THRESH
    diagnostics = {
        "source": "checkpoint",
        "thresholds": {
            "voiced": VOICED_THRESH,
            "breath": BREATH_THRESH,
            "onset": ONSET_THRESH,
        },
        "voiced_probability": {
            **_array_summary(voiced_np),
            "near_threshold_fraction": _near_threshold_fraction(voiced_np, VOICED_THRESH),
            "near_threshold_margin": 0.05,
        },
        "pitch_confidence": {
            "max_softmax_probability": _array_summary(pitch_conf_np),
            "top1_top2_margin": _array_summary(pitch_margin_np),
            "entropy": _array_summary(pitch_entropy_np),
            "normalized_entropy": _array_summary(
                pitch_entropy_np / max(np.log(float(N_BINS)), 1e-12)
            ),
        },
        "onset_probability": _array_summary(onset_np),
        "breath_probability": _array_summary(breath_np),
    }

    # Technique: prefer acoustic classifier when available
    acoustic_clf = _load_acoustic_classifier(ckpt_path, dev)
    if acoustic_clf is not None:
        acou_np  = extract_acoustic_features(vad_feats, pitch_hz, voiced_bool.astype(np.float32))
        acou_t   = torch.from_numpy(acou_np).unsqueeze(0).to(dev)
        with torch.no_grad():
            clip_repr, _ = model.encode_clip(hcqt_t, vad_t)
            tech_logits  = acoustic_clf(clip_repr, acou_t)
        source = "acoustic"
    else:
        tech_logits = tech_logits_base
        source = "model"

    tech_probs = torch.softmax(tech_logits[0], dim=-1).cpu().numpy()
    tech_idx   = int(np.argmax(tech_probs))

    return _build_result(
        audio, pitch_hz, voiced_bool, breath_bool, onset_bool,
        onset_raw_prob=onset_np,
        diagnostics=diagnostics,
        task_config=task_config,
        technique=TECHNIQUE_VOCAB[tech_idx],
        technique_confidence=float(tech_probs[tech_idx]),
        all_technique_scores={t: float(tech_probs[i]) for i, t in enumerate(TECHNIQUE_VOCAB)},
    )


# ---------------------------------------------------------------------------
# Fallback path (librosa.pyin)
# ---------------------------------------------------------------------------

def _run_fallback(
    audio: np.ndarray,
    task_config: TaskConfig | dict[str, Any] | None = None,
) -> CoachingResult:
    import librosa

    vad_ext   = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)
    vad_feats = vad_ext.compute(audio)

    f0_hz, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=float(FMIN), fmax=2100.0,
        sr=SR, hop_length=HOP_LENGTH,
        frame_length=2048,
        fill_na=0.0,
    )

    T         = min(vad_feats.shape[1], len(f0_hz))
    f0_hz     = f0_hz[:T].astype(np.float32)
    vad_feats = vad_feats[:, :T]
    voiced_probs = np.asarray(voiced_probs, dtype=np.float32)[:T]
    voiced_bool = f0_hz > 0

    breath_arr = derive_breath_labels(voiced_bool.astype(np.float32), vad_feats)
    onset_arr  = derive_onset_labels(f0_hz)

    vp_mean = float(np.mean(voiced_probs)) if len(voiced_probs) else 0.0
    vp_near = float(np.mean((voiced_probs > 0.35) & (voiced_probs < 0.65))) if len(voiced_probs) else 0.0

    return _build_result(
        audio, f0_hz, voiced_bool,
        breath_frames=breath_arr > 0.5,
        onset_frames=onset_arr > 0.5,
        onset_raw_prob=onset_arr,
        diagnostics={
            "source": "fallback",
            "raw_model_probabilities_available": False,
            "note": "Fallback inference uses librosa.pyin and heuristic breath/onset labels.",
            "onset_probability": _array_summary(onset_arr),
            "voiced_probability": {
                "mean": vp_mean,
                "near_threshold_fraction": vp_near,
            },
        },
        task_config=task_config,
        technique="unknown",
        technique_confidence=0.0,
        all_technique_scores={t: 0.0 for t in TECHNIQUE_VOCAB},
    )


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result(
    audio:          np.ndarray,
    pitch_hz:       np.ndarray,
    voiced:         np.ndarray,
    breath_frames:  np.ndarray,
    onset_frames:   np.ndarray,
    onset_raw_prob: np.ndarray,
    diagnostics:    dict | None,
    task_config:    TaskConfig | dict[str, Any] | None,
    technique:      str,
    technique_confidence: float,
    all_technique_scores: dict[str, float],
) -> CoachingResult:
    # ── Frame-level metrics ────────────────────────────────────────────────
    pitch_acc   = _pitch_accuracy(pitch_hz, voiced)
    drift_cents = _pitch_drift_cents(pitch_hz, voiced)
    phrases     = _phrase_lengths_s(voiced, HOP_S)
    breath_cnt  = int(np.diff(breath_frames.astype(np.int8), prepend=0).clip(min=0).sum())
    onset_cnt   = int(np.diff(onset_frames.astype(np.int8), prepend=0).clip(min=0).sum())
    onset_clar  = _onset_clarity(onset_raw_prob, onset_frames)

    # ── Algorithmic enrichments ────────────────────────────────────────────
    notes, coaching_pitch_hz, note_post_diag = segment_notes_for_coaching(pitch_hz, HOP_S)
    vq         = extract_voice_quality(audio, SR)
    vib_stats  = vibrato_summary(notes)
    diagnostics = _with_derived_diagnostics(
        diagnostics or {},
        pitch_hz=coaching_pitch_hz,
        voiced=coaching_pitch_hz > 0,
        notes=notes,
        onset_count=onset_cnt,
    )
    diagnostics["note_postprocessing"] = note_post_diag

    # ── Coaching text ──────────────────────────────────────────────────────
    score, summary, issues, exercises = _build_coaching_text(
        pitch_acc, drift_cents, phrases, onset_clar, onset_cnt,
        technique, technique_confidence,
        notes=notes, voice_quality=vq, vib_stats=vib_stats,
    )
    diagnostics.setdefault("raw_model_outputs", {})
    diagnostics["raw_model_outputs"].update(
        {
            "technique": technique,
            "technique_confidence": technique_confidence,
        }
    )
    analysis_validity = _analysis_validity(
        audio=audio,
        pitch_hz=pitch_hz,
        voiced=voiced,
        diagnostics=diagnostics,
        note_count=len(notes),
        onset_count=onset_cnt,
    )
    resolved_task_config = _resolve_task_config(task_config, analysis_validity)
    (
        score,
        full_song_score,
        diagnostic_score,
        score_status,
        score_caveat,
        summary,
        issues,
        exercises,
        technique,
        technique_confidence,
        task_analysis,
    ) = _apply_task_aware_scoring(
        score=score,
        original_full_song_score=score,
        summary=summary,
        issues=issues,
        exercises=exercises,
        technique=technique,
        technique_confidence=technique_confidence,
        analysis_validity=analysis_validity,
        task_config=resolved_task_config,
        diagnostics=diagnostics,
        pitch_drift_cents=drift_cents,
        onset_clarity=onset_clar,
        phrase_lengths_s=phrases,
    )

    return CoachingResult(
        pitch_hz=pitch_hz, voiced=voiced,
        breath_frames=breath_frames, onset_frames=onset_frames,
        hop_s=HOP_S,
        technique=technique,
        technique_confidence=technique_confidence,
        all_technique_scores=all_technique_scores,
        pitch_accuracy=pitch_acc,
        pitch_drift_cents=drift_cents,
        phrase_lengths_s=phrases,
        breath_count=breath_cnt,
        onset_count=onset_cnt,
        onset_clarity=onset_clar,
        notes=notes,
        voice_quality=vq,
        vibrato_stats=vib_stats,
        diagnostics=diagnostics,
        analysis_validity=analysis_validity,
        task_config=asdict(resolved_task_config),
        task_analysis=task_analysis,
        score=score,
        full_song_score=full_song_score,
        diagnostic_score=diagnostic_score,
        score_status=score_status,
        score_caveat=score_caveat,
        summary=summary,
        issues=issues, exercises=exercises,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _array_summary(values: np.ndarray) -> dict[str, float | dict[str, float] | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "percentiles": {},
        }
    percentiles = {
        f"p{p:02d}": float(np.percentile(arr, p))
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": percentiles,
    }


def _near_threshold_fraction(
    values: np.ndarray,
    threshold: float,
    margin: float = 0.05,
) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr - threshold) <= margin))


def _with_derived_diagnostics(
    diagnostics: dict,
    *,
    pitch_hz: np.ndarray,
    voiced: np.ndarray,
    notes: list[NoteSegment],
    onset_count: int,
) -> dict:
    out = dict(diagnostics)
    pitch = np.asarray(pitch_hz, dtype=np.float64)
    voiced_pitch = pitch[np.asarray(voiced, dtype=bool) & (pitch > 0)]
    duration_s = float(len(pitch_hz) * HOP_S)
    voiced_duration_s = float(len(voiced_pitch) * HOP_S)

    if voiced_pitch.size:
        out["f0"] = {
            "median_hz": float(np.median(voiced_pitch)),
            "mean_hz": float(np.mean(voiced_pitch)),
            "full_range_hz": {
                "min": float(np.min(voiced_pitch)),
                "max": float(np.max(voiced_pitch)),
            },
            "trimmed_range_hz": {
                "p05": float(np.percentile(voiced_pitch, 5)),
                "p95": float(np.percentile(voiced_pitch, 95)),
            },
            "low_frequency_threshold_hz": 80.0,
            "low_frequency_f0_ratio": float(np.mean(voiced_pitch < 80.0)),
        }
    else:
        out["f0"] = {
            "median_hz": None,
            "mean_hz": None,
            "full_range_hz": {"min": None, "max": None},
            "trimmed_range_hz": {"p05": None, "p95": None},
            "low_frequency_threshold_hz": 80.0,
            "low_frequency_f0_ratio": 0.0,
        }

    jumps = _f0_jump_metrics(pitch_hz)
    out["f0_jumps"] = {
        **jumps,
        "octave_jump_rate_per_second": (
            float(jumps["octave_jump_count"] / duration_s) if duration_s > 0 else 0.0
        ),
        "semitone_jump_rate_per_second": (
            float(jumps["semitone_jump_count"] / duration_s) if duration_s > 0 else 0.0
        ),
    }

    note_durations = np.asarray([n.duration_s for n in notes], dtype=np.float64)
    out["note_fragmentation"] = {
        "note_count": int(len(notes)),
        "notes_per_second": float(len(notes) / duration_s) if duration_s > 0 else 0.0,
        "notes_per_voiced_second": (
            float(len(notes) / voiced_duration_s) if voiced_duration_s > 0 else 0.0
        ),
        "median_note_duration_s": (
            float(np.median(note_durations)) if note_durations.size else None
        ),
        "short_note_ratio_lt_200ms": (
            float(np.mean(note_durations < 0.20)) if note_durations.size else 0.0
        ),
        "short_note_ratio_lt_300ms": (
            float(np.mean(note_durations < 0.30)) if note_durations.size else 0.0
        ),
        "onset_count": int(onset_count),
        "onsets_per_second": float(onset_count / duration_s) if duration_s > 0 else 0.0,
    }
    return out


def _f0_jump_metrics(pitch_hz: np.ndarray) -> dict[str, float | int]:
    f0 = np.asarray(pitch_hz, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid):
        return {
            "voiced_transition_count": 0,
            "octave_jump_count": 0,
            "semitone_jump_count": 0,
            "max_abs_semitone_jump": 0.0,
            "median_abs_semitone_jump": 0.0,
        }
    prev = f0[:-1][pair_valid]
    nxt = f0[1:][pair_valid]
    jumps = np.abs(12.0 * np.log2(nxt / prev.clip(min=1e-6)))
    return {
        "voiced_transition_count": int(jumps.size),
        "octave_jump_count": int(np.sum(jumps >= 12.0)),
        "semitone_jump_count": int(np.sum(jumps >= 1.5)),
        "max_abs_semitone_jump": float(np.max(jumps)),
        "median_abs_semitone_jump": float(np.median(jumps)),
    }


def _analysis_validity(
    *,
    audio: np.ndarray,
    pitch_hz: np.ndarray,
    voiced: np.ndarray,
    diagnostics: dict,
    note_count: int,
    onset_count: int,
) -> dict:
    duration_s = float(len(pitch_hz) * HOP_S)
    audio_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    voiced_duration_s = float(np.sum(voiced) * HOP_S)
    vad_mean = _nested_float(diagnostics, "voiced_probability.mean", 0.0)
    vad_near = _nested_float(diagnostics, "voiced_probability.near_threshold_fraction", 0.0)
    pitch_conf = _nested_float(
        diagnostics,
        "pitch_confidence.max_softmax_probability.mean",
        0.0,
    )
    pitch_margin = _nested_float(
        diagnostics,
        "pitch_confidence.top1_top2_margin.mean",
        0.0,
    )
    pitch_entropy = _nested_float(
        diagnostics,
        "pitch_confidence.normalized_entropy.mean",
        1.0,
    )
    f0_p05 = _nested_float(diagnostics, "f0.trimmed_range_hz.p05", 0.0)
    f0_p95 = _nested_float(diagnostics, "f0.trimmed_range_hz.p95", 0.0)
    trimmed_range = max(0.0, f0_p95 - f0_p05)
    low_f0_ratio = _nested_float(diagnostics, "f0.low_frequency_f0_ratio", 0.0)
    octave_rate = _nested_float(diagnostics, "f0_jumps.octave_jump_rate_per_second", 0.0)
    semitone_rate = _nested_float(diagnostics, "f0_jumps.semitone_jump_rate_per_second", 0.0)
    raw_note_count = _nested_float(diagnostics, "note_postprocessing.raw_note_count", float(note_count))
    raw_fragmentation = _nested_float(
        diagnostics,
        "note_postprocessing.raw_fragmentation_index",
        0.0,
    )
    raw_octave_count = _nested_float(diagnostics, "note_postprocessing.octave_jump_count", 0.0)
    raw_semitone_count = _nested_float(diagnostics, "note_postprocessing.semitone_jump_count", 0.0)
    raw_octave_rate = raw_octave_count / duration_s if duration_s > 0 else 0.0
    raw_semitone_rate = raw_semitone_count / duration_s if duration_s > 0 else 0.0
    notes_per_second = _nested_float(
        diagnostics,
        "note_fragmentation.notes_per_second",
        0.0,
    )
    short_note_ratio = _nested_float(
        diagnostics,
        "note_fragmentation.short_note_ratio_lt_300ms",
        0.0,
    )
    onsets_per_second = float(onset_count / duration_s) if duration_s > 0 else 0.0

    metrics = {
        "duration_s": duration_s,
        "audio_rms": audio_rms,
        "voiced_frame_ratio": voiced_ratio,
        "voiced_duration_s": voiced_duration_s,
        "voiced_probability_mean": vad_mean,
        "voiced_probability_near_threshold_fraction": vad_near,
        "pitch_confidence_mean": pitch_conf,
        "pitch_confidence_margin_mean": pitch_margin,
        "pitch_normalized_entropy_mean": pitch_entropy,
        "f0_trimmed_range_hz": trimmed_range,
        "low_frequency_f0_ratio": low_f0_ratio,
        "octave_jump_rate_per_second": octave_rate,
        "semitone_jump_rate_per_second": semitone_rate,
        "raw_note_count": int(raw_note_count),
        "raw_fragmentation_index": raw_fragmentation,
        "raw_octave_jump_rate_per_second": raw_octave_rate,
        "raw_semitone_jump_rate_per_second": raw_semitone_rate,
        "note_count": int(note_count),
        "notes_per_second": notes_per_second,
        "short_note_ratio_lt_300ms": short_note_ratio,
        "onset_count": int(onset_count),
        "onsets_per_second": onsets_per_second,
    }

    reason_codes: list[str] = []
    input_type = "analyzable_singing"
    confidence = 0.75
    is_fallback = diagnostics.get("source") == "fallback"

    if duration_s < 1.0 or voiced_duration_s < 0.5:
        input_type = "low_confidence_or_unreliable"
        reason_codes.append("too_little_voiced_audio")
        confidence = 0.80
    elif (
        audio_rms < 0.005
        and (is_fallback or (pitch_conf < 0.18 and pitch_margin < 0.02 and pitch_entropy > 0.60))
    ):
        input_type = "no_voice_or_noise"
        reason_codes.extend([
            "very_low_audio_rms",
        ])
        if not is_fallback:
            reason_codes.extend(["low_pitch_confidence", "high_pitch_entropy"])
        if vad_near > 0.20:
            reason_codes.append("voiced_probabilities_near_threshold")
        confidence = 0.90
    elif not is_fallback and pitch_conf < 0.16 and pitch_margin < 0.02 and pitch_entropy > 0.62:
        input_type = "low_confidence_or_unreliable"
        reason_codes.extend(["low_pitch_confidence", "high_pitch_entropy"])
        confidence = 0.75
    elif (
        voiced_ratio > 0.95
        and duration_s >= 2.0
        and (
            (trimmed_range < 140.0 and note_count >= 2)
            or raw_note_count >= 8
            or raw_octave_rate > 2.0
        )
    ):
        input_type = "diagnostic_sustained_tone"
        reason_codes.append("continuous_voicing")
        if trimmed_range < 140.0:
            reason_codes.append("limited_trimmed_f0_range")
        if raw_note_count >= 8 or semitone_rate > 2.0:
            reason_codes.append("fragmented_f0_tracking")
        confidence = 0.70
    elif (
        voiced_ratio > 0.90
        and trimmed_range > 80.0
        and note_count <= 4
        and notes_per_second < 0.80
        and raw_note_count <= 5
    ):
        input_type = "diagnostic_pitch_slide"
        reason_codes.extend(["continuous_voicing", "wide_f0_movement", "few_note_events"])
        confidence = 0.72
    elif (
        not is_fallback
        and pitch_conf < 0.25
        and (notes_per_second > 1.2 or raw_fragmentation > 1.5)
        and (octave_rate > 1.0 or raw_octave_rate > 1.0 or short_note_ratio > 0.50)
    ):
        input_type = "speech_like_or_non_singing"
        reason_codes.extend([
            "speech_like_fragmentation",
            "low_pitch_confidence",
        ])
        if octave_rate > 1.0 or raw_octave_rate > 1.0:
            reason_codes.append("frequent_octave_jumps")
        confidence = 0.72
    else:
        if not reason_codes:
            reason_codes.append("passes_current_postprocessing_checks")
        if not is_fallback and pitch_conf < 0.30:
            reason_codes.append("melody_scoring_low_pitch_confidence_caveat")
            confidence = 0.62

    is_analyzable = input_type == "analyzable_singing"
    if input_type.startswith("diagnostic_"):
        is_analyzable = False

    return {
        "is_analyzable": is_analyzable,
        "input_type": input_type,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason_codes": reason_codes,
        "summary_metrics": metrics,
    }


def _resolve_task_config(
    task_config: TaskConfig | dict[str, Any] | None,
    analysis_validity: dict,
) -> TaskConfig:
    if isinstance(task_config, TaskConfig):
        cfg = task_config
    elif isinstance(task_config, dict):
        cfg = TaskConfig(
            task_type=str(task_config.get("task_type") or "free_singing"),
            skill_focus=task_config.get("skill_focus"),
            target=task_config.get("target"),
            reference=task_config.get("reference"),
            scoring_mode=str(task_config.get("scoring_mode") or "auto"),
            strictness=task_config.get("strictness"),
        )
    else:
        input_type = analysis_validity.get("input_type")
        inferred = {
            "diagnostic_sustained_tone": "sustained_note",
            "diagnostic_pitch_slide": "pitch_slide",
        }.get(input_type, "free_singing")
        cfg = TaskConfig(task_type=inferred, scoring_mode="auto")
    if cfg.task_type not in TASK_TYPES:
        cfg.task_type = "free_singing"
    return cfg


def _apply_task_aware_scoring(
    *,
    score: int | None,
    original_full_song_score: int,
    summary: str,
    issues: list[str],
    exercises: list[str],
    technique: str,
    technique_confidence: float,
    analysis_validity: dict,
    task_config: TaskConfig,
    diagnostics: dict,
    pitch_drift_cents: float,
    onset_clarity: float,
    phrase_lengths_s: list[float],
) -> tuple[int | None, int | None, int | None, str, str | None, str, list[str], list[str], str, float, dict]:
    input_type = analysis_validity.get("input_type", "low_confidence_or_unreliable")
    task_type = task_config.task_type
    task_analysis = {
        "task_type": task_type,
        "provided_task_config": asdict(task_config),
        "detected_input_type": input_type,
        "caveats": [],
        "status": "not_scored",
        "summary": "",
        "scoring_components": {},
    }

    if input_type in {
        "no_voice_or_noise",
        "speech_like_or_non_singing",
        "low_confidence_or_unreliable",
    }:
        neutral_messages = {
            "no_voice_or_noise": "No analyzable singing was detected.",
            "speech_like_or_non_singing": (
                "This sounds like speech or non-singing voice, so singing coaching was not generated."
            ),
            "low_confidence_or_unreliable": (
                "The audio was too noisy or unreliable to score confidently."
            ),
        }
        score_status = {
            "no_voice_or_noise": "no_analyzable_singing",
            "speech_like_or_non_singing": "speech_or_non_singing_no_score",
            "low_confidence_or_unreliable": "low_confidence_no_score",
        }[input_type]
        task_analysis.update(
            {
                "status": score_status,
                "summary": neutral_messages[input_type],
                "caveats": ["Task scoring skipped because input was not analyzable singing."],
            }
        )
        return (
            None,
            None,
            None,
            score_status,
            None,
            neutral_messages[input_type],
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type == "sustained_note":
        diagnostic_score, components = _sustained_note_score(
            analysis_validity,
            pitch_drift_cents,
        )
        summary = "Sustained-note diagnostic complete; full-song scoring was not generated."
        caveat = "Diagnostic sustained-note score only; no reference melody was evaluated."
        task_analysis.update(
            {
                "status": "diagnostic_sustained_tone_only",
                "summary": summary,
                "caveats": [caveat],
                "scoring_components": components,
            }
        )
        return (
            diagnostic_score,
            None,
            diagnostic_score,
            "diagnostic_sustained_tone_only",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type == "pitch_slide":
        diagnostic_score, components = _pitch_slide_score(analysis_validity, diagnostics)
        summary = "Pitch-slide diagnostic complete; full-song scoring was not generated."
        caveat = "Diagnostic pitch-slide score only; no reference melody was evaluated."
        task_analysis.update(
            {
                "status": "diagnostic_pitch_slide_only",
                "summary": summary,
                "caveats": [caveat],
                "scoring_components": components,
            }
        )
        return (
            diagnostic_score,
            None,
            diagnostic_score,
            "diagnostic_pitch_slide_only",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type == "reference_song":
        caveat = "Reference-song scoring is not implemented yet; no reference melody comparison was performed."
        summary = "Reference-song task received, but reference melody scoring is not available yet."
        task_analysis.update(
            {
                "status": "reference_scoring_not_implemented",
                "summary": summary,
                "caveats": [caveat],
            }
        )
        return (
            None,
            None,
            None,
            "reference_scoring_not_implemented",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type in {"scale", "interval"}:
        target = task_config.target or {}
        target_notes = target.get("notes") or target.get("target_notes")
        if not target_notes:
            caveat = "Target notes are required for scale or interval scoring."
            summary = f"{task_type.replace('_', ' ').title()} task needs target notes before scoring."
            task_analysis.update(
                {
                    "status": "insufficient_target_info",
                    "summary": summary,
                    "caveats": [caveat],
                }
            )
            return (
                None,
                None,
                None,
                "insufficient_target_info",
                caveat,
                summary,
                [],
                [],
                "not_applicable",
                0.0,
                task_analysis,
            )
        diagnostic_score, components = _scale_interval_score(analysis_validity, target_notes)
        summary = f"{task_type.replace('_', ' ').title()} diagnostic complete using detected note movement."
        caveat = "Scale/interval scoring is provisional and uses detected f0 movement only."
        task_analysis.update(
            {
                "status": f"{task_type}_provisional",
                "summary": summary,
                "caveats": [caveat],
                "scoring_components": components,
            }
        )
        return (
            diagnostic_score,
            None,
            diagnostic_score,
            f"{task_type}_provisional",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type == "rhythm":
        diagnostic_score, components = _rhythm_score(onset_clarity, analysis_validity)
        caveat = "Rhythm scoring is preliminary unless a reference beat or timing grid is provided."
        summary = "Rhythm diagnostic complete using detected onset activity."
        task_analysis.update(
            {
                "status": "rhythm_provisional_no_reference_grid",
                "summary": summary,
                "caveats": [caveat],
                "scoring_components": components,
            }
        )
        return (
            diagnostic_score,
            None,
            diagnostic_score,
            "rhythm_provisional_no_reference_grid",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type in {"breath_control", "tone_consistency"}:
        diagnostic_score, components = _breath_tone_score(
            task_type,
            analysis_validity,
            phrase_lengths_s,
        )
        summary = f"{task_type.replace('_', ' ').title()} diagnostic complete."
        caveat = "Diagnostic score only; full-song reference scoring was not generated."
        task_analysis.update(
            {
                "status": f"{task_type}_diagnostic_only",
                "summary": summary,
                "caveats": [caveat],
                "scoring_components": components,
            }
        )
        return (
            diagnostic_score,
            None,
            diagnostic_score,
            f"{task_type}_diagnostic_only",
            caveat,
            summary,
            [],
            [],
            "not_applicable",
            0.0,
            task_analysis,
        )

    if task_type == "free_singing":
        caveat = "Score is based on detected pitch and timing features, not a reference melody."
        if "reference melody" not in summary:
            summary = summary + f" Note: {caveat}"
        task_analysis.update(
            {
                "status": "free_singing_general_feedback",
                "summary": summary,
                "caveats": [caveat],
            }
        )
        return (
            original_full_song_score,
            original_full_song_score,
            None,
            "free_singing_general_feedback",
            caveat,
            summary,
            issues,
            exercises,
            technique,
            technique_confidence,
            task_analysis,
        )

    # Safe fallback for unknown task types after normalization.
    summary = "Task could not be scored with the current evaluator."
    task_analysis.update({"status": "unsupported_task", "summary": summary})
    return (
        None,
        None,
        None,
        "unsupported_task",
        "Unsupported task type.",
        summary,
        [],
        [],
        "not_applicable",
        0.0,
        task_analysis,
    )


def _sustained_note_score(
    analysis_validity: dict,
    pitch_drift_cents: float,
) -> tuple[int, dict]:
    metrics = analysis_validity.get("summary_metrics") or {}
    voiced_ratio = float(metrics.get("voiced_frame_ratio") or 0.0)
    semitone_rate = float(metrics.get("semitone_jump_rate_per_second") or 0.0)
    short_ratio = float(metrics.get("short_note_ratio_lt_300ms") or 0.0)
    fragmentation = float(metrics.get("notes_per_second") or 0.0)
    drift_abs = abs(float(pitch_drift_cents))
    continuity_score = np.clip(voiced_ratio, 0.0, 1.0) * 30.0
    stability_score = max(0.0, 1.0 - semitone_rate / 6.0) * 25.0
    drift_score = max(0.0, 1.0 - drift_abs / 75.0) * 20.0
    dropout_score = max(0.0, 1.0 - (1.0 - voiced_ratio) / 0.25) * 15.0
    fragmentation_score = max(0.0, 1.0 - max(0.0, fragmentation - 0.25) / 2.0) * 10.0
    components = {
        "voicing_continuity": float(continuity_score),
        "pitch_stability": float(stability_score),
        "pitch_drift": float(drift_score),
        "dropout": float(dropout_score),
        "fragmentation": float(fragmentation_score),
        "pitch_drift_cents_abs": drift_abs,
        "voiced_frame_ratio": voiced_ratio,
        "short_note_ratio_lt_300ms": short_ratio,
    }
    return int(round(np.clip(sum(v for k, v in components.items() if k in {
        "voicing_continuity", "pitch_stability", "pitch_drift", "dropout", "fragmentation"
    }), 0, 100))), components


def _pitch_slide_score(
    analysis_validity: dict,
    diagnostics: dict,
) -> tuple[int, dict]:
    metrics = analysis_validity.get("summary_metrics") or {}
    voiced_ratio = float(metrics.get("voiced_frame_ratio") or 0.0)
    trimmed_range = float(metrics.get("f0_trimmed_range_hz") or 0.0)
    semitone_rate = float(metrics.get("semitone_jump_rate_per_second") or 0.0)
    note_count = float(metrics.get("note_count") or 0.0)
    full_min = _nested_float(diagnostics, "f0.full_range_hz.min", 0.0)
    full_max = _nested_float(diagnostics, "f0.full_range_hz.max", 0.0)
    direction = "up" if full_max >= full_min else "unknown"
    smoothness_score = max(0.0, 1.0 - semitone_rate / 4.0) * 30.0
    range_score = min(1.0, trimmed_range / 160.0) * 25.0
    continuity_score = np.clip(voiced_ratio, 0.0, 1.0) * 25.0
    note_count_score = max(0.0, 1.0 - max(0.0, note_count - 3.0) / 6.0) * 10.0
    direction_score = 10.0 if direction in {"up", "down"} else 5.0
    components = {
        "slide_smoothness": float(smoothness_score),
        "range": float(range_score),
        "continuity": float(continuity_score),
        "few_note_fragments": float(note_count_score),
        "direction": direction,
        "direction_score": direction_score,
        "f0_trimmed_range_hz": trimmed_range,
        "voiced_frame_ratio": voiced_ratio,
    }
    total = smoothness_score + range_score + continuity_score + note_count_score + direction_score
    return int(round(np.clip(total, 0, 100))), components


def _scale_interval_score(
    analysis_validity: dict,
    target_notes: Any,
) -> tuple[int, dict]:
    metrics = analysis_validity.get("summary_metrics") or {}
    note_count = float(metrics.get("note_count") or 0.0)
    expected = len(target_notes) if isinstance(target_notes, list) else 1
    coverage = min(1.0, note_count / max(expected, 1))
    score = int(round(coverage * 60.0))
    return score, {
        "detected_note_count": note_count,
        "target_note_count": expected,
        "coverage": coverage,
    }


def _rhythm_score(onset_clarity: float, analysis_validity: dict) -> tuple[int, dict]:
    metrics = analysis_validity.get("summary_metrics") or {}
    onset_rate = float(metrics.get("onsets_per_second") or 0.0)
    activity = min(1.0, onset_rate / 2.0)
    clarity = min(1.0, max(0.0, onset_clarity))
    score = int(round((clarity * 0.65 + activity * 0.35) * 100.0))
    return score, {
        "onset_clarity": float(onset_clarity),
        "onsets_per_second": onset_rate,
        "activity_score": activity,
    }


def _breath_tone_score(
    task_type: str,
    analysis_validity: dict,
    phrase_lengths_s: list[float],
) -> tuple[int, dict]:
    metrics = analysis_validity.get("summary_metrics") or {}
    voiced_ratio = float(metrics.get("voiced_frame_ratio") or 0.0)
    mean_phrase = float(np.mean(phrase_lengths_s)) if phrase_lengths_s else 0.0
    stability = max(0.0, 1.0 - float(metrics.get("semitone_jump_rate_per_second") or 0.0) / 6.0)
    if task_type == "breath_control":
        phrase_score = min(1.0, mean_phrase / 6.0) * 60.0
        continuity_score = voiced_ratio * 40.0
        total = phrase_score + continuity_score
        components = {
            "mean_phrase_s": mean_phrase,
            "phrase_score": float(phrase_score),
            "continuity_score": float(continuity_score),
        }
    else:
        continuity_score = voiced_ratio * 40.0
        stability_score = stability * 60.0
        total = continuity_score + stability_score
        components = {
            "continuity_score": float(continuity_score),
            "stability_score": float(stability_score),
        }
    return int(round(np.clip(total, 0, 100))), components


def _nested_float(obj: dict, dotted_path: str, default: float) -> float:
    cur = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    if cur is None:
        return default
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _pitch_accuracy(pitch_hz: np.ndarray, voiced: np.ndarray) -> float:
    if not voiced.any():
        return 0.0
    f = pitch_hz[voiced]
    f = f[f > 0]
    if len(f) == 0:
        return 0.0
    nearest = 440.0 * 2.0 ** (np.round(12.0 * np.log2(f / 440.0)) / 12.0)
    cents   = 1200.0 * np.log2(f / nearest.clip(min=1e-6))
    return float((np.abs(cents) < 50.0).mean())


def _pitch_drift_cents(pitch_hz: np.ndarray, voiced: np.ndarray) -> float:
    if not voiced.any():
        return 0.0
    f = pitch_hz[voiced]
    f = f[f > 0]
    if len(f) == 0:
        return 0.0
    nearest = 440.0 * 2.0 ** (np.round(12.0 * np.log2(f / 440.0)) / 12.0)
    cents   = 1200.0 * np.log2(f / nearest.clip(min=1e-6))
    return float(np.median(cents))


def _phrase_lengths_s(voiced: np.ndarray, hop_s: float) -> list[float]:
    min_silence_frames = max(1, round(MIN_SILENCE_S / hop_s))
    min_phrase_frames  = max(1, round(MIN_PHRASE_S  / hop_s))
    phrases = []
    v = voiced.astype(np.int8)
    i, n = 0, len(v)
    while i < n:
        if not v[i]:
            i += 1
            continue
        start       = i
        silence_run = 0
        while i < n:
            if v[i]:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    break
            i += 1
        end = i - silence_run
        if (end - start) >= min_phrase_frames:
            phrases.append((end - start) * hop_s)
    return phrases


def _onset_clarity(onset_prob: np.ndarray, onset_frames: np.ndarray) -> float:
    if onset_frames.any():
        return float(onset_prob[onset_frames].mean())
    return float(onset_prob.mean())


# ---------------------------------------------------------------------------
# Coaching text — rule-based, enriched by algorithmic signals
# ---------------------------------------------------------------------------

def _build_coaching_text(
    pitch_accuracy:    float,
    pitch_drift_cents: float,
    phrase_lengths_s:  list[float],
    onset_clarity:     float,
    onset_count:       int,
    technique:         str,
    technique_confidence: float,
    notes:             list[NoteSegment],
    voice_quality:     VoiceQuality | None,
    vib_stats:         dict,
) -> tuple[int, str, list[str], list[str]]:
    issues:    list[str] = []
    exercises: list[str] = []

    mean_phrase = float(np.mean(phrase_lengths_s)) if phrase_lengths_s else 0.0

    # ── 1. Pitch accuracy ─────────────────────────────────────────────────
    if pitch_accuracy < 0.60:
        issues.append(
            f"Pitch needs significant work — only {pitch_accuracy:.0%} of notes "
            "were in tune."
        )
        exercises.append(
            "Sing slowly to a drone note at A4 (440 Hz). Match the pitch "
            "and hold it for 4 s before moving on."
        )
    elif pitch_accuracy < 0.80:
        issues.append(
            f"Pitch accuracy is {pitch_accuracy:.0%} — work on landing "
            "the centre of each note."
        )
        exercises.append(
            "Long-tone practice: sustain one note for 4 s, record yourself, "
            "compare to a reference pitch app."
        )

    # ── 2. Per-note pitch problems (from note segmentation) ───────────────
    problem_notes = flat_notes_summary(notes, threshold_cents=30.0)
    if problem_notes:
        note_list = ", ".join(problem_notes[:3])
        direction = "flat" if any("flat" in s for s in problem_notes) else "sharp"
        issues.append(
            f"Specific notes are consistently off: {note_list}."
        )
        if direction == "flat":
            exercises.append(
                "For those notes, imagine lifting the back of your tongue slightly "
                "and 'thinking' the pitch higher before you sing it."
            )
        else:
            exercises.append(
                "For those notes, relax the jaw and let the breath drop lower in "
                "the body before onset to prevent over-shooting."
            )

    # ── 3. Pitch drift ────────────────────────────────────────────────────
    if pitch_drift_cents < -25 and not any("flat" in i for i in issues):
        issues.append(
            f"Overall tendency to sing flat ({pitch_drift_cents:+.0f} ¢ median). "
            "Flat singing sounds unintentionally sad or unclear."
        )
        exercises.append(
            "Raise your soft palate ('bright' vowels) and think the pitch "
            "slightly higher than it feels comfortable."
        )
    elif pitch_drift_cents > 25 and not any("sharp" in i for i in issues):
        issues.append(
            f"Overall tendency to sing sharp ({pitch_drift_cents:+.0f} ¢ median). "
            "Too much air pressure often causes sharpness."
        )
        exercises.append(
            "Relax jaw tension and take deeper, lower breaths. "
            "Let the sound settle rather than pushing it out."
        )

    # ── 4. Voice quality (parselmouth) ────────────────────────────────────
    if voice_quality is not None:
        if voice_quality.breathiness == "breathy":
            issues.append(
                f"Breathy voice quality detected (HNR {voice_quality.hnr_db:.0f} dB — "
                "air escaping before the cords fully close)."
            )
            exercises.append(
                "Hum 'mmm' with lips lightly closed to build cord closure and "
                "forward resonance. Gradually open to 'mah' while keeping the buzz."
            )
        elif voice_quality.breathiness == "mild" and voice_quality.hnr_db < 15:
            issues.append(
                f"Slightly airy tone (HNR {voice_quality.hnr_db:.0f} dB). "
                "More cord engagement will give a clearer, more projected sound."
            )
            exercises.append(
                "Try 'staccato' vowel exercises (short, clear 'ha-ha-ha') to "
                "encourage full cord closure at each attack."
            )

        if voice_quality.is_unstable:
            jit = voice_quality.jitter_pct
            shi = voice_quality.shimmer_pct
            issues.append(
                f"Vocal instability detected (jitter {jit:.1f}%, shimmer {shi:.1f}%). "
                "This can indicate tension, fatigue, or insufficient warm-up."
            )
            exercises.append(
                "Rest the voice for 20 min, hydrate well, then warm up with "
                "gentle lip trills before attempting full-voice singing."
            )

    # ── 5. Breath support / phrase length ─────────────────────────────────
    if mean_phrase < 3.5:
        issues.append(
            f"Phrases average only {mean_phrase:.1f} s — breath runs out too quickly."
        )
        exercises.append(
            "Diaphragmatic breathing: inhale silently for 4 counts, then sustain "
            "'sss' for 8 counts. Build up to 12 counts over a week."
        )
    elif mean_phrase < 5.0:
        issues.append(
            f"Phrase length averages {mean_phrase:.1f} s. "
            "Aim for 5–6 s to improve musical line."
        )
        exercises.append(
            "Before each phrase take a fuller breath and see how long you can "
            "sustain 'aaah' on a comfortable note. Target: 6 s without strain."
        )

    # ── 6. Vibrato ────────────────────────────────────────────────────────
    n_long = vib_stats.get("n_long_notes", 0)
    n_vib  = vib_stats.get("n_vibrato_notes", 0)
    rate   = vib_stats.get("mean_rate_hz", 0.0)
    depth  = vib_stats.get("mean_depth_cents", 0.0)

    if n_long >= 2 and n_vib == 0:
        issues.append(
            f"{n_long} sustained note(s) detected but no vibrato found. "
            "Adding vibrato enriches long notes and shows vocal control."
        )
        exercises.append(
            "Practice a gentle hand-on-chest pulse while sustaining a note — "
            "this helps initiate the natural 5–6 Hz oscillation of healthy vibrato."
        )
    elif n_vib > 0 and (rate < 4.5 or rate > 7.5):
        issues.append(
            f"Vibrato detected but rate is {rate:.1f} Hz "
            f"(healthy range: 5–7 Hz). {'Too slow' if rate < 4.5 else 'Too fast'} "
            "vibrato sounds unnatural."
        )
        exercises.append(
            "Practise sustaining notes with a metronome set to 6 Hz (360 bpm "
            "subdivisions) and aim to match your oscillation to that pulse."
        )
    elif n_vib > 0 and depth < 25:
        issues.append(
            f"Vibrato is present but very shallow ({depth:.0f} ¢ depth). "
            "Aim for 30–60 ¢ for a warm, full sound."
        )
        exercises.append(
            "Allow more jaw relaxation on long notes and let the air stream "
            "have natural pulse — don't 'lock' the sound."
        )

    # ── 7. Onset clarity ──────────────────────────────────────────────────
    if onset_clarity < 0.38 and onset_count > 2:
        issues.append(
            f"{onset_count} note attack(s) sound hesitant or under-powered "
            f"(onset clarity {onset_clarity:.2f})."
        )
        exercises.append(
            "Practice 'ha' onsets: place a hand on your belly and feel it push "
            "outward at each attack. Sharp, confident starts."
        )

    # Cap at 4 issues (prioritise pitch, voice quality, breath, vibrato)
    issues    = issues[:4]
    exercises = exercises[:4]

    # ── Score ─────────────────────────────────────────────────────────────
    phrase_support = min(1.0, mean_phrase / 6.0) if phrase_lengths_s else 0.5
    vq_score = 1.0
    if voice_quality is not None:
        vq_score = min(1.0, max(0.0, (voice_quality.hnr_db - 5.0) / 20.0))

    score = round(
        pitch_accuracy * 50
        + vq_score      * 20
        + min(onset_clarity, 1.0) * 15
        + phrase_support * 15
    )
    score = max(0, min(100, score))

    # ── Summary ───────────────────────────────────────────────────────────
    if score >= 80:
        vib_note = f" Vibrato detected on {n_vib} note(s)." if n_vib > 0 else ""
        quality_desc = 'clear' if (voice_quality and voice_quality.breathiness == 'clear') else 'good'
        summary = (
            f"Excellent singing! Pitch {pitch_accuracy:.0%} accurate, "
            f"voice quality {quality_desc}."
            + vib_note
        )
    elif score >= 65:
        summary = (
            f"Good foundation — pitch {pitch_accuracy:.0%}, "
            f"avg phrase {mean_phrase:.1f} s. "
            "The issues below will make a clear difference."
        )
    elif score >= 50:
        summary = (
            f"Solid effort. Pitch {pitch_accuracy:.0%}, score {score}/100. "
            "Pitch accuracy and breath support are your priorities."
        )
    else:
        summary = (
            f"Keep practising! Score {score}/100. "
            "Focus on one issue at a time — start with staying in tune."
        )

    return score, summary, issues, exercises


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _print_report(result: CoachingResult, *, use_colour: bool = True) -> None:
    R  = "\033[0m"  if use_colour else ""
    B  = "\033[1m"  if use_colour else ""
    G  = "\033[32m" if use_colour else ""
    Y  = "\033[33m" if use_colour else ""
    RE = "\033[31m" if use_colour else ""
    C  = "\033[36m" if use_colour else ""

    display_score = result.score
    sc_col = G if (display_score or 0) >= 80 else (Y if (display_score or 0) >= 60 else RE)

    dur_s      = len(result.pitch_hz) * result.hop_s
    mean_phrase = float(np.mean(result.phrase_lengths_s)) if result.phrase_lengths_s else 0.0
    vq         = result.voice_quality
    vs         = result.vibrato_stats

    print()
    print(f"{B}{'━' * 60}{R}")
    print(f"{B}  VocalStars Coaching Report{R}")
    print(f"{'━' * 60}")
    print(f"  Duration      : {dur_s:.1f} s  ({len(result.pitch_hz)} frames @ 10 ms)")
    print(f"  Technique     : {result.technique}  ({result.technique_confidence:.0%} confident)")
    print(f"  Pitch acc     : {result.pitch_accuracy:.1%}  |  Drift: {result.pitch_drift_cents:+.0f} ¢")
    print(f"  Notes detected: {len(result.notes)}"
          f"  |  Phrases: {len(result.phrase_lengths_s)} @ avg {mean_phrase:.1f} s")
    print(f"  Breaths       : {result.breath_count}"
          f"  |  Onset clarity: {result.onset_clarity:.2f}")

    if vq is not None:
        print(f"  Voice quality : HNR {vq.hnr_db:.1f} dB"
              f"  jitter {vq.jitter_pct:.1f}%"
              f"  shimmer {vq.shimmer_pct:.1f}%"
              f"  → {vq.breathiness}")

    n_vib = vs.get("n_vibrato_notes", 0)
    n_long = vs.get("n_long_notes", 0)
    if n_long > 0:
        vib_txt = (f"{n_vib}/{n_long} notes with vibrato "
                   f"({vs['mean_rate_hz']:.1f} Hz, {vs['mean_depth_cents']:.0f} ¢)"
                   if n_vib > 0 else f"no vibrato on {n_long} long note(s)")
        print(f"  Vibrato       : {vib_txt}")

    print(f"{'━' * 60}")
    if display_score is None:
        print(f"  {B}Score: {Y}not produced ({result.score_status}){R}")
    else:
        print(f"  {B}Score: {sc_col}{display_score}/100{R}  ({result.score_status})")
    print(f"  {result.summary}")

    if result.issues:
        print()
        print(f"{B}  Issues & Exercises:{R}")
        for i, (issue, ex) in enumerate(zip(result.issues, result.exercises), 1):
            print(f"\n  {C}{i}. {issue}{R}")
            print(f"     {Y}→ {ex}{R}")

    print(f"\n{'━' * 60}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ml_new.inference.coach_inference",
        description="Analyse a singing recording and print coaching feedback.",
    )
    parser.add_argument("audio", help="Path to the audio file (.wav, .mp3, …)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Path to .pt checkpoint (default: ml_new/checkpoints/unified/best.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="cpu | mps | cuda")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output raw JSON instead of formatted report")
    args = parser.parse_args(argv)

    ckpt = args.checkpoint
    if ckpt is None:
        default = _ROOT / "ml_new" / "checkpoints" / "unified" / "best.pt"
        if default.exists():
            ckpt = str(default)

    result = analyse_recording(args.audio, checkpoint=ckpt, device=args.device)

    if args.json:
        out = asdict(result)
        def _serialise(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            return v
        # Recursively convert numpy types
        import copy
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            return _serialise(obj)
        print(json.dumps(_clean(out), indent=2))
    else:
        _print_report(result, use_colour=sys.stdout.isatty())


if __name__ == "__main__":
    main()
