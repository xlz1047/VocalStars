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
from ml_new.inference.algorithms import (
    NoteSegment, VoiceQuality, VibratoInfo,
    segment_notes, extract_voice_quality,
    flat_notes_summary, vibrato_summary,
)

SR         = 16_000
HOP_LENGTH = 160
HOP_S: float = HOP_LENGTH / SR   # 0.01 s per frame
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

    # ── Human-readable coaching output ───────────────────────────────────
    score:     int         # 0–100 overall beginner score
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
# Model inference path
# ---------------------------------------------------------------------------

def _run_model(audio: np.ndarray, ckpt_path: Path, device: str) -> CoachingResult:
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

    with torch.no_grad():
        pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, _ = model(
            torch.from_numpy(hcqt).unsqueeze(0).to(dev),
            torch.from_numpy(vad_feats).unsqueeze(0).to(dev),
        )

    bin_hz_np  = model.bin_hz.cpu().numpy()
    pitch_hz   = bin_hz_np[pitch_logits[0].argmax(dim=-1).cpu().numpy()].astype(np.float32)
    voiced_np  = voiced_prob[0].cpu().numpy()
    breath_np  = breath_prob[0].cpu().numpy()
    onset_np   = onset_prob[0].cpu().numpy()

    voiced_bool = voiced_np >= VOICED_THRESH
    pitch_hz    = np.where(voiced_bool, pitch_hz, 0.0).astype(np.float32)
    breath_bool = breath_np >= BREATH_THRESH
    onset_bool  = onset_np  >= ONSET_THRESH

    tech_probs = torch.softmax(tech_logits[0], dim=-1).cpu().numpy()
    tech_idx   = int(np.argmax(tech_probs))

    return _build_result(
        audio, pitch_hz, voiced_bool, breath_bool, onset_bool,
        onset_raw_prob=onset_np,
        technique=TECHNIQUE_VOCAB[tech_idx],
        technique_confidence=float(tech_probs[tech_idx]),
        all_technique_scores={t: float(tech_probs[i]) for i, t in enumerate(TECHNIQUE_VOCAB)},
    )


# ---------------------------------------------------------------------------
# Fallback path (librosa.pyin)
# ---------------------------------------------------------------------------

def _run_fallback(audio: np.ndarray) -> CoachingResult:
    import librosa

    vad_ext   = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)
    vad_feats = vad_ext.compute(audio)

    f0_hz, _, _ = librosa.pyin(
        audio, fmin=float(FMIN), fmax=2100.0,
        sr=SR, hop_length=HOP_LENGTH, fill_na=0.0,
    )

    T         = min(vad_feats.shape[1], len(f0_hz))
    f0_hz     = f0_hz[:T].astype(np.float32)
    vad_feats = vad_feats[:, :T]
    voiced_bool = f0_hz > 0

    breath_arr = derive_breath_labels(voiced_bool.astype(np.float32), vad_feats)
    onset_arr  = derive_onset_labels(f0_hz)

    return _build_result(
        audio, f0_hz, voiced_bool,
        breath_frames=breath_arr > 0.5,
        onset_frames=onset_arr > 0.5,
        onset_raw_prob=onset_arr,
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
    notes      = segment_notes(pitch_hz, HOP_S)
    vq         = extract_voice_quality(audio, SR)
    vib_stats  = vibrato_summary(notes)

    # ── Coaching text ──────────────────────────────────────────────────────
    score, summary, issues, exercises = _build_coaching_text(
        pitch_acc, drift_cents, phrases, onset_clar, onset_cnt,
        technique, technique_confidence,
        notes=notes, voice_quality=vq, vib_stats=vib_stats,
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
        score=score, summary=summary,
        issues=issues, exercises=exercises,
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
        summary = (
            f"Excellent singing! Pitch {pitch_accuracy:.0%} accurate, "
            f"voice quality {"clear" if (voice_quality and voice_quality.breathiness == "clear") else "good"}."
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

    sc_col = G if result.score >= 80 else (Y if result.score >= 60 else RE)

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
    print(f"  {B}Score: {sc_col}{result.score}/100{R}")
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
