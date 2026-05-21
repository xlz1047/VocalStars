from typing import Any
import numpy as np
import librosa


def extract_pitch_features(audio_path: str) -> dict[str, Any]:
    """Estimate fundamental frequency contour and simple stability metrics.

    Uses librosa.pyin (a robust F0 estimator) when available. Returns a
    lightweight summary suitable for coaching heuristics.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Use librosa.pyin to estimate f0; fallback to empty if it fails.
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        # Replace nan with 0 for stability computations
        f0_clean = np.nan_to_num(f0, nan=0.0)
        voiced_ratio = float(np.sum(~np.isnan(f0)) / max(1, len(f0)))
        pitch_curve = f0_clean.tolist()
        stability_score = float(np.std(f0_clean[f0_clean > 0])) if np.any(f0_clean > 0) else 0.0
    except Exception:
        pitch_curve = []
        voiced_ratio = 0.0
        stability_score = 0.0

    # Very simple note segmentation by detecting voiced segments
    estimated_notes = []
    try:
        frames = librosa.times_like(f0, sr=sr)
        current_note = None
        for t, f in zip(frames, f0):
            if f is not None and not np.isnan(f):
                if current_note is None:
                    current_note = {"start": t, "pitches": [float(f)]}
                else:
                    current_note["pitches"].append(float(f))
            else:
                if current_note is not None:
                    current_note["end"] = t
                    # rough median pitch
                    current_note["median_pitch_hz"] = float(np.median(current_note["pitches"]))
                    estimated_notes.append(current_note)
                    current_note = None
        if current_note is not None:
            current_note["end"] = frames[-1] if len(frames) else 0.0
            current_note["median_pitch_hz"] = float(np.median(current_note["pitches"]))
            estimated_notes.append(current_note)
    except Exception:
        estimated_notes = []

    return {
        "stability_score": stability_score,
        "voiced_ratio": voiced_ratio,
        "estimated_notes": estimated_notes,
        "pitch_curve": pitch_curve,
    }
