from typing import Any
import librosa
import numpy as np


def analyze_rhythm(audio_path: str) -> dict[str, Any]:
    """Estimate tempo and basic timing variance.

    Uses librosa.beat to estimate tempo and onsets to compute timing jitter.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    try:
        # Primary: use beat_track which also returns beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # If beat_track fails to find tempo, fallback to onset envelope + tempo estimation
        if not tempo or float(tempo) == 0.0:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            tempo = float(tempos[0]) if len(tempos) else 0.0
            # approximate beat times from tempo
            beat_times = np.arange(0, len(y) / sr, 60.0 / max(tempo, 1.0))
        else:
            beat_times = librosa.frames_to_time(beats, sr=sr)

        ibis = np.diff(beat_times) if len(beat_times) > 1 else np.array([])
        timing_variance = float(np.std(ibis)) if ibis.size else 0.0
        beat_alignment = "aligned" if timing_variance < 0.05 else "loose"
    except Exception:
        tempo = None
        timing_variance = None
        beat_alignment = "unknown"

    return {"tempo": tempo, "timing_variance": timing_variance, "beat_alignment": beat_alignment}
