"""Coaching inference module for VocalStars.

Takes a raw audio file and returns a CoachingResult with:
  - Per-frame pitch, voicing, breath and onset arrays
  - Clip-level technique classification
  - Derived metrics (pitch accuracy, drift, phrase lengths, etc.)
  - Human-readable score, summary, issues and exercises

Usage
-----
    from ml_new.inference.coach_inference import analyse_recording
    result = analyse_recording("my_singing.wav",
                               checkpoint="ml_new/checkpoints/unified/best.pt")
    print(result.summary)
    for issue, ex in zip(result.issues, result.exercises):
        print(f"  Issue: {issue}")
        print(f"  Try:   {ex}")

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
from dataclasses import dataclass, asdict, field
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

SR = 16_000
HOP_LENGTH = 160
HOP_S: float = HOP_LENGTH / SR          # 0.01 s per frame
N_BINS = 180
BINS_PER_OCTAVE = 36
FMIN = UnifiedVocalModel.FMIN           # 32.7 Hz (C1)

VOICED_THRESH = 0.50
BREATH_THRESH = 0.35
ONSET_THRESH  = 0.30

# Phrases shorter than this are ignored (fragments / micro-pauses)
MIN_PHRASE_S = 0.5
# Unvoiced gap must be this long to split a phrase
MIN_SILENCE_S = 0.10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoachingResult:
    """Full coaching analysis for one recorded clip."""

    # Raw per-frame arrays (10 ms / frame resolution)
    pitch_hz:      np.ndarray  # float32, 0.0 = unvoiced
    voiced:        np.ndarray  # bool
    breath_frames: np.ndarray  # bool
    onset_frames:  np.ndarray  # bool
    hop_s:         float       # seconds per frame (always 0.01)

    # Clip-level technique
    technique:            str
    technique_confidence: float
    all_technique_scores: dict[str, float]

    # Derived coaching metrics
    pitch_accuracy:    float       # fraction of voiced frames within 50 ¢ of nearest semitone
    pitch_drift_cents: float       # median signed cents offset (+ = sharp, − = flat)
    phrase_lengths_s:  list[float] # seconds per phrase
    breath_count:      int
    onset_count:       int
    onset_clarity:     float       # mean onset probability at detected onset frames

    # Human-readable coaching output
    score:     int         # 0–100 overall beginner score
    summary:   str         # one-sentence overview
    issues:    list[str]   # up to 3 detected problems
    exercises: list[str]   # one targeted exercise per issue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_recording(
    audio_path: str | Path,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> CoachingResult:
    """Analyse a singing recording and return coaching feedback.

    Args:
        audio_path: Path to any audio file supported by librosa (.wav, .mp3, …).
        checkpoint: Path to a trained ``UnifiedVocalModel`` ``.pt`` checkpoint.
            When ``None`` or the file is missing, falls back to librosa.pyin
            for pitch and heuristic breath/onset labelling.
        device: PyTorch device string (``"cpu"``, ``"mps"``, ``"cuda"``).

    Returns:
        :class:`CoachingResult` populated with raw arrays, derived metrics and
        human-readable coaching advice.
    """
    import librosa

    audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)

    ckpt_path = Path(checkpoint) if checkpoint else None
    use_model = (ckpt_path is not None) and ckpt_path.exists()

    if use_model:
        result = _run_model(audio, ckpt_path, device)
    else:
        result = _run_fallback(audio)

    return result


# ---------------------------------------------------------------------------
# Model-based inference path
# ---------------------------------------------------------------------------

def _run_model(audio: np.ndarray, ckpt_path: Path, device: str) -> CoachingResult:
    hcqt_ext = HCQTExtractor(
        sr=SR, hop_length=HOP_LENGTH,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    )
    vad_ext = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)

    hcqt      = hcqt_ext.compute(audio)       # (6, 180, T)
    vad_feats = vad_ext.compute(audio)         # (3, T)

    T = min(hcqt.shape[2], vad_feats.shape[1])
    hcqt      = hcqt[:, :, :T]
    vad_feats = vad_feats[:, :T]

    dev = torch.device(device)
    model = UnifiedVocalModel().to(dev)
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    hcqt_t    = torch.from_numpy(hcqt).unsqueeze(0).to(dev)      # (1,6,180,T)
    vad_t     = torch.from_numpy(vad_feats).unsqueeze(0).to(dev)  # (1,3,T)

    with torch.no_grad():
        pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, _ = \
            model(hcqt_t, vad_t)

    # Pitch: argmax decode to Hz
    pitch_bins = pitch_logits[0].argmax(dim=-1).cpu().numpy()  # (T,)
    bin_hz_np  = model.bin_hz.cpu().numpy()
    pitch_hz   = bin_hz_np[pitch_bins].astype(np.float32)

    voiced_np  = voiced_prob[0].cpu().numpy()
    breath_np  = breath_prob[0].cpu().numpy()
    onset_np   = onset_prob[0].cpu().numpy()

    voiced_bool = voiced_np >= VOICED_THRESH
    # Zero out pitch where not voiced
    pitch_hz = np.where(voiced_bool, pitch_hz, 0.0).astype(np.float32)

    breath_bool = breath_np >= BREATH_THRESH
    onset_bool  = onset_np  >= ONSET_THRESH

    # Technique
    tech_probs_np = torch.softmax(tech_logits[0], dim=-1).cpu().numpy()
    tech_idx      = int(np.argmax(tech_probs_np))
    technique     = TECHNIQUE_VOCAB[tech_idx]
    tech_conf     = float(tech_probs_np[tech_idx])
    all_scores    = {t: float(tech_probs_np[i]) for i, t in enumerate(TECHNIQUE_VOCAB)}

    return _build_result(
        pitch_hz, voiced_bool, breath_bool, onset_bool,
        onset_raw_prob=onset_np,
        technique=technique,
        technique_confidence=tech_conf,
        all_technique_scores=all_scores,
    )


# ---------------------------------------------------------------------------
# Fallback path (no checkpoint — uses librosa.pyin)
# ---------------------------------------------------------------------------

def _run_fallback(audio: np.ndarray) -> CoachingResult:
    import librosa

    vad_ext   = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)
    vad_feats = vad_ext.compute(audio)         # (3, T)

    f0_hz, voiced_flag, _ = librosa.pyin(
        audio, fmin=float(FMIN), fmax=2100.0,
        sr=SR, hop_length=HOP_LENGTH,
        fill_na=0.0,
    )

    T = min(vad_feats.shape[1], len(f0_hz))
    f0_hz     = f0_hz[:T].astype(np.float32)
    vad_feats = vad_feats[:, :T]
    voiced_bool = (f0_hz > 0)

    vad_label  = voiced_bool.astype(np.float32)
    breath_arr = derive_breath_labels(vad_label, vad_feats)
    onset_arr  = derive_onset_labels(f0_hz)

    breath_bool  = breath_arr > 0.5
    onset_bool   = onset_arr  > 0.5
    onset_raw    = onset_arr  # 0/1 heuristic labels

    return _build_result(
        f0_hz, voiced_bool, breath_bool, onset_bool,
        onset_raw_prob=onset_raw,
        technique="unknown",
        technique_confidence=0.0,
        all_technique_scores={t: 0.0 for t in TECHNIQUE_VOCAB},
    )


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result(
    pitch_hz:       np.ndarray,
    voiced:         np.ndarray,
    breath_frames:  np.ndarray,
    onset_frames:   np.ndarray,
    onset_raw_prob: np.ndarray,
    technique:      str,
    technique_confidence: float,
    all_technique_scores: dict[str, float],
) -> CoachingResult:
    pitch_acc   = _pitch_accuracy(pitch_hz, voiced)
    drift_cents = _pitch_drift_cents(pitch_hz, voiced)
    phrases     = _phrase_lengths_s(voiced, HOP_S)
    breath_cnt  = int(np.diff(breath_frames.astype(np.int8), prepend=0).clip(min=0).sum())
    # Count rising-edge events, not raw frames (consecutive onset frames = 1 note attack)
    onset_cnt   = int(np.diff(onset_frames.astype(np.int8), prepend=0).clip(min=0).sum())
    onset_clar  = _onset_clarity(onset_raw_prob, onset_frames)

    score, summary, issues, exercises = _build_coaching_text(
        pitch_acc, drift_cents, phrases, onset_clar,
        onset_cnt, technique, technique_confidence,
    )

    return CoachingResult(
        pitch_hz=pitch_hz,
        voiced=voiced,
        breath_frames=breath_frames,
        onset_frames=onset_frames,
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
        score=score,
        summary=summary,
        issues=issues,
        exercises=exercises,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pitch_accuracy(pitch_hz: np.ndarray, voiced: np.ndarray) -> float:
    if not voiced.any():
        return 0.0
    f = pitch_hz[voiced]
    f = f[f > 0]
    if len(f) == 0:
        return 0.0
    # Nearest equal-temperament semitone (A4 = 440 Hz reference)
    nearest = 440.0 * 2.0 ** (np.round(12.0 * np.log2(f / 440.0)) / 12.0)
    cents = 1200.0 * np.log2(f / nearest.clip(min=1e-6))
    return float((np.abs(cents) < 50.0).mean())


def _pitch_drift_cents(pitch_hz: np.ndarray, voiced: np.ndarray) -> float:
    if not voiced.any():
        return 0.0
    f = pitch_hz[voiced]
    f = f[f > 0]
    if len(f) == 0:
        return 0.0
    nearest = 440.0 * 2.0 ** (np.round(12.0 * np.log2(f / 440.0)) / 12.0)
    cents = 1200.0 * np.log2(f / nearest.clip(min=1e-6))
    return float(np.median(cents))


def _phrase_lengths_s(voiced: np.ndarray, hop_s: float) -> list[float]:
    min_silence_frames = max(1, round(MIN_SILENCE_S / hop_s))
    min_phrase_frames  = max(1, round(MIN_PHRASE_S  / hop_s))

    phrases = []
    v = voiced.astype(np.int8)
    i = 0
    n = len(v)
    while i < n:
        if not v[i]:
            i += 1
            continue
        start = i
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
        length_frames = end - start
        if length_frames >= min_phrase_frames:
            phrases.append(length_frames * hop_s)

    return phrases


def _onset_clarity(onset_prob: np.ndarray, onset_frames: np.ndarray) -> float:
    """Mean onset probability at detected onset peaks (global mean if none)."""
    if onset_frames.any():
        return float(onset_prob[onset_frames].mean())
    return float(onset_prob.mean())


# ---------------------------------------------------------------------------
# Coaching text generation
# ---------------------------------------------------------------------------

def _build_coaching_text(
    pitch_accuracy:    float,
    pitch_drift_cents: float,
    phrase_lengths_s:  list[float],
    onset_clarity:     float,
    onset_count:       int,
    technique:         str,
    technique_confidence: float,
) -> tuple[int, str, list[str], list[str]]:
    issues:    list[str] = []
    exercises: list[str] = []

    # ── Pitch accuracy ────────────────────────────────────────────────────
    if pitch_accuracy < 0.60:
        issues.append(
            f"Pitch needs significant work — only {pitch_accuracy:.0%} of notes "
            "were in tune. Many notes are off-key."
        )
        exercises.append(
            "Sing very slowly to a drone note at A4 (440 Hz). Match the pitch "
            "and hold it for 4 seconds before moving to the next note."
        )
    elif pitch_accuracy < 0.80:
        issues.append(
            f"Pitch accuracy is {pitch_accuracy:.0%} — focus on landing the "
            "centre of each note more consistently."
        )
        exercises.append(
            "Practice long tones: sustain a single note for 4 seconds while "
            "listening carefully. Record yourself and compare to a reference pitch."
        )

    # ── Pitch drift ───────────────────────────────────────────────────────
    if pitch_drift_cents < -20:
        issues.append(
            f"You tend to sing flat (avg {abs(pitch_drift_cents):.0f} ¢ below pitch). "
            "Flat singing can sound dull or sad unintentionally."
        )
        exercises.append(
            "Raise your soft palate and think 'bright' as you sustain notes. "
            "Imagining the sound coming from behind your eyes can help lift pitch."
        )
    elif pitch_drift_cents > 20:
        issues.append(
            f"You tend to sing sharp (avg {pitch_drift_cents:.0f} ¢ above pitch). "
            "Over-pushing air often causes sharpness."
        )
        exercises.append(
            "Relax jaw tension and take deeper, lower breaths. "
            "Let the sound drop into the chest rather than pushing up."
        )

    # ── Breath support / phrase length ────────────────────────────────────
    mean_phrase = float(np.mean(phrase_lengths_s)) if phrase_lengths_s else 0.0
    if mean_phrase < 3.5:
        issues.append(
            f"Phrases average only {mean_phrase:.1f} s — you may be running out "
            "of breath. Short phrases limit musical expression."
        )
        exercises.append(
            "Practice diaphragmatic breathing: inhale silently for 4 counts, "
            "then sustain 'sss' for 8 counts. Repeat daily to build breath capacity."
        )
    elif mean_phrase < 5.0:
        issues.append(
            f"Phrase length averages {mean_phrase:.1f} s. "
            "Try extending phrases to 5–6 s for better musical flow."
        )
        exercises.append(
            "Take a deeper breath before each phrase and see how long you can "
            "sustain 'aaah' on a comfortable note. Aim for 6 s before inhaling."
        )

    # ── Onset clarity ─────────────────────────────────────────────────────
    if onset_clarity < 0.40 and onset_count > 0:
        issues.append(
            f"{onset_count} note attacks detected but onset clarity is low — "
            "note starts sound hesitant or under-powered."
        )
        exercises.append(
            "Practice clean 'ha' onsets: start each note with a gentle aspirate. "
            "Place your hand on your stomach and feel it push outward at each 'ha'."
        )

    # ── Technique-specific advice ─────────────────────────────────────────
    if technique == "breathy" and technique_confidence > 0.50:
        issues.append(
            f"Breathy tone detected ({technique_confidence:.0%} confidence). "
            "Air is escaping before the vocal cords fully close."
        )
        exercises.append(
            "Hum on 'mmm' with lips lightly closed to build forward resonance "
            "and improve cord closure. Gradually open to 'mah' while keeping the buzz."
        )
    elif technique == "vocal_fry" and technique_confidence > 0.50:
        issues.append(
            f"Vocal fry detected ({technique_confidence:.0%} confidence). "
            "Creak at note starts or ends strains the voice over time."
        )
        exercises.append(
            "Warm up gently with lip trills (brrr) before singing. "
            "Stay hydrated and avoid fry deliberately — start notes on a clean vowel."
        )

    # Cap at 3 issues
    issues    = issues[:3]
    exercises = exercises[:3]

    # ── Score ─────────────────────────────────────────────────────────────
    phrase_support = min(1.0, mean_phrase / 6.0) if phrase_lengths_s else 0.5
    score = round(pitch_accuracy * 60 + min(onset_clarity, 1.0) * 20 + phrase_support * 20)
    score = max(0, min(100, score))

    # ── Summary ───────────────────────────────────────────────────────────
    if score >= 80:
        summary = (
            f"Excellent singing! Pitch is {pitch_accuracy:.0%} accurate with good breath support. "
            "Small refinements will take you to the next level."
        )
    elif score >= 65:
        summary = (
            f"Good foundation — pitch accuracy is {pitch_accuracy:.0%}. "
            "Focusing on the issues below will make a clear difference."
        )
    elif score >= 50:
        summary = (
            f"Solid effort with a score of {score}/100. "
            "Work on pitch and breath support as priorities."
        )
    else:
        summary = (
            f"Keep practising! Score: {score}/100. "
            "Focus on one issue at a time — start with staying in tune."
        )

    return score, summary, issues, exercises


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(result: CoachingResult, *, use_colour: bool = True) -> None:
    """Print a human-readable coaching report."""
    RESET  = "\033[0m"  if use_colour else ""
    BOLD   = "\033[1m"  if use_colour else ""
    GREEN  = "\033[32m" if use_colour else ""
    YELLOW = "\033[33m" if use_colour else ""
    RED    = "\033[31m" if use_colour else ""
    CYAN   = "\033[36m" if use_colour else ""

    def score_colour(s: int) -> str:
        if s >= 80: return GREEN
        if s >= 60: return YELLOW
        return RED

    dur_s = len(result.pitch_hz) * result.hop_s
    print()
    print(f"{BOLD}{'━' * 56}{RESET}")
    print(f"{BOLD}  VocalStars Coaching Report{RESET}")
    print(f"{'━' * 56}")
    print(f"  Duration     : {dur_s:.1f} s   |  {len(result.pitch_hz)} frames @ 10 ms")
    print(f"  Technique    : {result.technique}  ({result.technique_confidence:.0%} confident)")
    print(f"  Pitch acc    : {result.pitch_accuracy:.1%}"
          f"  |  Drift: {result.pitch_drift_cents:+.0f} ¢")
    mean_phrase = float(np.mean(result.phrase_lengths_s)) if result.phrase_lengths_s else 0.0
    print(f"  Phrases      : {len(result.phrase_lengths_s)} phrase(s), "
          f"avg {mean_phrase:.1f} s")
    print(f"  Breaths      : {result.breath_count}")
    print(f"  Onsets       : {result.onset_count}  |  Clarity: {result.onset_clarity:.2f}")
    print(f"{'━' * 56}")
    sc = result.score
    print(f"  {BOLD}Score: {score_colour(sc)}{sc}/100{RESET}")
    print(f"  {result.summary}")
    if result.issues:
        print()
        print(f"{BOLD}  Issues & Exercises:{RESET}")
        for i, (issue, ex) in enumerate(zip(result.issues, result.exercises), 1):
            print(f"\n  {CYAN}{i}. {issue}{RESET}")
            print(f"     {YELLOW}→ {ex}{RESET}")
    print(f"\n{'━' * 56}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ml_new.inference.coach_inference",
        description="Analyse a singing recording and print coaching feedback.",
    )
    parser.add_argument("audio", help="Path to the audio file (.wav, .mp3, …)")
    parser.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to trained UnifiedVocalModel .pt checkpoint. "
             "Defaults to ml_new/checkpoints/unified/best.pt",
    )
    parser.add_argument(
        "--device", "-d", default="cpu",
        help="PyTorch device: cpu | mps | cuda (default: cpu)",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args(argv)

    ckpt = args.checkpoint
    if ckpt is None:
        default = _ROOT / "ml_new" / "checkpoints" / "unified" / "best.pt"
        if default.exists():
            ckpt = str(default)

    result = analyse_recording(args.audio, checkpoint=ckpt, device=args.device)

    if args.json:
        out = asdict(result)
        # Convert numpy arrays to lists for JSON serialisation
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
        print(json.dumps(out, indent=2))
    else:
        _print_report(result, use_colour=sys.stdout.isatty())


if __name__ == "__main__":
    main()
