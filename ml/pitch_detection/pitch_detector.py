from typing import Any


def extract_pitch_features(audio_path: str) -> dict[str, Any]:
    """Placeholder for pitch detection using librosa, torchcrepe, or Parselmouth."""
    # TODO: load waveform and extract pitch contours from audio.
    return {
        "stability_score": None,
        "estimated_notes": [],
        "pitch_curve": [],
        "note_transitions": [],
    }
