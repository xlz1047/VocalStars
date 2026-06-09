"""Signal-processing algorithms for VocalStars coaching.

Three independent, data-free analysers that run on top of the model's output:

1. Note segmentation  — splits the F0 contour into individual note events,
   giving per-note pitch accuracy and stability.

2. Voice quality      — HNR, jitter, shimmer via parselmouth (PRAAT) to assess
   breathiness, instability, and vocal fatigue.  More reliable than the neural
   technique head at this stage.

3. Vibrato analysis   — per-note F0 autocorrelation to detect vibrato rate and
   depth on sustained notes (>600 ms).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

HOP_S: float = 0.01  # seconds per frame (10 ms)

# ── Note naming ───────────────────────────────────────────────────────────────
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_note_name(hz: float) -> str:
    """Convert a frequency in Hz to a note name like 'A4' or 'C#3'."""
    if hz <= 0:
        return "?"
    semitones_from_a4 = round(12.0 * np.log2(hz / 440.0))
    semitones_from_c0 = semitones_from_a4 + 57       # A4 is 57 semitones above C0
    octave    = semitones_from_c0 // 12
    note_idx  = semitones_from_c0 % 12
    return f"{_NOTE_NAMES[note_idx]}{octave}"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class VibratoInfo:
    rate_hz:      float  # oscillation frequency (5–7 Hz = healthy)
    depth_cents:  float  # peak-to-peak amplitude in cents (30–70¢ = healthy)
    regularity:   float  # 0–1, autocorrelation peak height (> 0.6 = regular)


@dataclass
class NoteSegment:
    start_s:          float       # onset time (seconds)
    duration_s:       float       # note length (seconds)
    pitch_hz:         float       # mean fundamental frequency
    note_name:        str         # e.g. "A4"
    cents_error:      float       # signed deviation from nearest semitone (+ = sharp)
    stability_cents:  float       # std dev of cents within the note
    vibrato:          VibratoInfo | None = None  # present if vibrato detected


@dataclass
class VoiceQuality:
    hnr_db:       float   # Harmonics-to-Noise Ratio — higher = cleaner/more modal
    jitter_pct:   float   # Local pitch perturbation (%) — < 1 % = stable
    shimmer_pct:  float   # Local amplitude perturbation (%) — < 4 % = stable

    # Interpretation thresholds (derived from literature / clinical norms)
    @property
    def breathiness(self) -> str:
        """'clear' | 'mild' | 'breathy'."""
        if self.hnr_db >= 20:
            return "clear"
        if self.hnr_db >= 12:
            return "mild"
        return "breathy"

    @property
    def is_unstable(self) -> bool:
        return self.jitter_pct > 1.0 or self.shimmer_pct > 4.0


# ── 1. Note segmentation ─────────────────────────────────────────────────────

def segment_notes(
    pitch_hz: np.ndarray,
    hop_s:    float = HOP_S,
    min_duration_s:       float = 0.10,
    jump_thresh_semitones: float = 1.5,
) -> list[NoteSegment]:
    """Segment an F0 contour into individual note events.

    A new note begins when:
    - Pitch becomes non-zero after silence (voiced onset after gap), OR
    - Pitch jumps by more than ``jump_thresh_semitones`` between adjacent frames.

    Args:
        pitch_hz: ``(T,)`` float32 array, 0.0 = unvoiced.
        hop_s: Seconds per frame.
        min_duration_s: Notes shorter than this are discarded.
        jump_thresh_semitones: Pitch discontinuity threshold for note splits.

    Returns:
        List of :class:`NoteSegment`, in chronological order.
    """
    jump_thresh_cents = jump_thresh_semitones * 100.0
    min_frames = max(1, round(min_duration_s / hop_s))

    notes: list[NoteSegment] = []
    voiced = pitch_hz > 0
    T = len(pitch_hz)

    i = 0
    while i < T:
        if not voiced[i]:
            i += 1
            continue

        seg_start = i
        seg_frames = [i]

        while i + 1 < T and voiced[i + 1]:
            i += 1
            prev_f = pitch_hz[i - 1]
            curr_f = pitch_hz[i]
            cents_jump = abs(1200.0 * np.log2(curr_f / max(prev_f, 1e-6)))
            if cents_jump > jump_thresh_cents:
                # Close current segment, start new one
                if len(seg_frames) >= min_frames:
                    notes.append(_make_note(seg_frames, pitch_hz, hop_s))
                seg_frames = [i]
            else:
                seg_frames.append(i)

        if len(seg_frames) >= min_frames:
            notes.append(_make_note(seg_frames, pitch_hz, hop_s))
        i += 1

    # Add vibrato analysis to each note in-place
    for note in notes:
        note.vibrato = _detect_vibrato_for_note(note, pitch_hz, hop_s)

    return notes


def _make_note(frames: list[int], pitch_hz: np.ndarray, hop_s: float) -> NoteSegment:
    f_arr     = pitch_hz[frames]
    mean_hz   = float(np.mean(f_arr))
    start_s   = frames[0] * hop_s
    dur_s     = len(frames) * hop_s

    nearest_hz = 440.0 * 2.0 ** (round(12.0 * np.log2(mean_hz / 440.0)) / 12.0)
    cents_err  = float(1200.0 * np.log2(mean_hz / max(nearest_hz, 1e-6)))

    # Stability: std of per-frame cents deviation from note mean
    per_frame_cents = 1200.0 * np.log2(f_arr / max(mean_hz, 1e-6))
    stability       = float(np.std(per_frame_cents))

    return NoteSegment(
        start_s=start_s,
        duration_s=dur_s,
        pitch_hz=mean_hz,
        note_name=hz_to_note_name(mean_hz),
        cents_error=cents_err,
        stability_cents=stability,
    )


# ── 2. Voice quality (parselmouth / PRAAT) ────────────────────────────────────

def extract_voice_quality(audio: np.ndarray, sr: int) -> VoiceQuality | None:
    """Compute HNR, local jitter, and local shimmer via PRAAT (parselmouth).

    Returns ``None`` if parselmouth is unavailable or the audio is too short.

    Interpretation:
        HNR (dB): > 20 clear, 12–20 mild breathiness, < 12 heavy breathiness/fry
        Jitter (%): < 1 % stable, 1–2 % borderline, > 2 % unstable
        Shimmer (%): < 3 % stable, 3–5 % borderline, > 5 % unstable
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None

    if len(audio) < sr * 0.5:  # need at least 0.5 s
        return None

    try:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))

        # HNR (Cross-Correlation method)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_db = float(call(harmonicity, "Get mean", 0, 0))
        if not np.isfinite(hnr_db):
            hnr_db = 0.0

        # Jitter and shimmer via pitch-synchronous PointProcess
        pitch_obj = call(snd, "To Pitch", 0.0, 75, 600)
        pp = call([snd, pitch_obj], "To PointProcess (cc)")

        jitter = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        if not np.isfinite(jitter):  jitter  = 0.0
        if not np.isfinite(shimmer): shimmer = 0.0

        return VoiceQuality(
            hnr_db=hnr_db,
            jitter_pct=jitter * 100.0,
            shimmer_pct=shimmer * 100.0,
        )
    except Exception:
        return None


# ── 3. Vibrato detection ──────────────────────────────────────────────────────

_VIBRATO_RATE_MIN_HZ = 4.5
_VIBRATO_RATE_MAX_HZ = 8.0
_MIN_VIBRATO_DEPTH_CENTS = 15.0   # peak-to-peak floor (¢)
_MAX_VIBRATO_DEPTH_CENTS = 120.0  # above this is glissando/arpeggio, not vibrato
_MIN_REGULARITY = 0.45            # autocorrelation peak threshold
# Notes with std > this are glissandos / arpeggios — skip vibrato detection
_MAX_NOTE_STABILITY_CENTS = 80.0


def _detect_vibrato_for_note(
    note: NoteSegment,
    pitch_hz: np.ndarray,
    hop_s: float,
) -> VibratoInfo | None:
    """Detect vibrato in a single note using F0 autocorrelation.

    Requires at least 600 ms (60 frames at 10 ms/frame) to resolve vibrato
    cycles reliably.

    Returns ``None`` if the note is too short or no vibrato is found.
    """
    min_frames = max(1, round(0.60 / hop_s))
    start_frame = round(note.start_s / hop_s)
    n_frames    = round(note.duration_s / hop_s)

    if n_frames < min_frames:
        return None

    # Guard: notes with large pitch variation are arpeggios / glissandos, not vibrato
    if note.stability_cents > _MAX_NOTE_STABILITY_CENTS:
        return None

    f0_seg = pitch_hz[start_frame: start_frame + n_frames]
    voiced = f0_seg > 0
    if voiced.sum() < min_frames:
        return None

    # Work in cents relative to note mean — removes absolute pitch, keeps modulation
    mean_f0 = float(np.mean(f0_seg[voiced]))
    cents = np.where(voiced, 1200.0 * np.log2(f0_seg / max(mean_f0, 1e-6)), 0.0)

    # Linear detrend to remove slow drift that would alias into vibrato frequencies
    cents = sp_signal.detrend(cents)

    # Autocorrelation (un-normalised sum, then normalised)
    ac = np.correlate(cents, cents, mode="full")
    ac = ac[len(cents) - 1:]      # keep positive lags only
    if ac[0] < 1e-10:
        return None
    ac = ac / ac[0]

    # Search for peak in vibrato lag range
    fps        = 1.0 / hop_s
    lag_min    = max(1, int(fps / _VIBRATO_RATE_MAX_HZ))
    lag_max    = int(fps / _VIBRATO_RATE_MIN_HZ)

    if lag_max >= len(ac):
        return None

    peak_offset = int(np.argmax(ac[lag_min: lag_max + 1]))
    peak_lag    = lag_min + peak_offset
    peak_val    = float(ac[peak_lag])

    if peak_val < _MIN_REGULARITY:
        return None

    rate_hz     = fps / peak_lag
    depth_cents = float(np.std(cents) * 2.0 * np.sqrt(2))  # ≈ peak-to-peak

    if depth_cents < _MIN_VIBRATO_DEPTH_CENTS:
        return None
    if depth_cents > _MAX_VIBRATO_DEPTH_CENTS:
        return None

    return VibratoInfo(
        rate_hz=rate_hz,
        depth_cents=depth_cents,
        regularity=peak_val,
    )


# ── Aggregate helpers used by coach_inference ─────────────────────────────────

def flat_notes_summary(notes: list[NoteSegment], threshold_cents: float = 25.0) -> list[str]:
    """Return up to 3 note names that are consistently flat or sharp.

    Args:
        notes: List of :class:`NoteSegment` from :func:`segment_notes`.
        threshold_cents: Minimum absolute cents error to flag a note.

    Returns:
        Short human-readable strings like ``"A4 (−38 ¢, flat)"`` sorted by
        absolute error descending.
    """
    from collections import defaultdict
    by_note: dict[str, list[float]] = defaultdict(list)
    for n in notes:
        if abs(n.cents_error) >= threshold_cents:
            by_note[n.note_name].append(n.cents_error)

    summaries = []
    for name, errs in by_note.items():
        if len(errs) < 2:  # only flag notes that are wrong repeatedly
            continue
        mean_err = float(np.mean(errs))
        direction = "flat" if mean_err < 0 else "sharp"
        summaries.append((abs(mean_err), f"{name} ({mean_err:+.0f} ¢, {direction})"))

    summaries.sort(reverse=True)
    return [s for _, s in summaries[:3]]


def vibrato_summary(notes: list[NoteSegment]) -> dict:
    """Summarise vibrato presence across all notes.

    Returns a dict with keys:
        ``n_long_notes``, ``n_vibrato_notes``,
        ``mean_rate_hz``, ``mean_depth_cents``.
    """
    min_vibrato_dur = 0.60
    long_notes    = [n for n in notes if n.duration_s >= min_vibrato_dur]
    vibrato_notes = [n for n in long_notes if n.vibrato is not None]

    if vibrato_notes:
        mean_rate  = float(np.mean([n.vibrato.rate_hz     for n in vibrato_notes]))
        mean_depth = float(np.mean([n.vibrato.depth_cents for n in vibrato_notes]))
    else:
        mean_rate  = 0.0
        mean_depth = 0.0

    return {
        "n_long_notes":    len(long_notes),
        "n_vibrato_notes": len(vibrato_notes),
        "mean_rate_hz":    mean_rate,
        "mean_depth_cents": mean_depth,
    }
