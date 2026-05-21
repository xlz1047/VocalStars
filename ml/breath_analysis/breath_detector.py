from typing import Any
import librosa
import numpy as np


def analyze_breath(audio_path: str) -> dict[str, Any]:
    """Estimate breath points and simple support metrics from energy envelope.

    This is a heuristic approach: compute the short-time energy and find
    local minima to suggest breath locations and variability.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    try:
        # Use silence-gap detection: find non-silent intervals and treat gaps as breaths
        # top_db chosen conservatively for synthetic signals; tune for real recordings
        non_silent = librosa.effects.split(y, top_db=30)
        breath_times = []
        min_gap_samples = int(0.06 * sr)
        for i in range(len(non_silent) - 1):
            gap_start = non_silent[i][1]
            gap_end = non_silent[i + 1][0]
            gap_len = gap_end - gap_start
            if gap_len >= min_gap_samples:
                center_sample = (gap_start + gap_end) // 2
                breath_times.append(float(center_sample) / sr)

        cycles = np.diff(breath_times) if len(breath_times) > 1 else np.array([])
        breath_length_variation = float(np.std(cycles)) if cycles.size else 0.0
        support_score = float(max(0.0, 1.0 - breath_length_variation))
    except Exception:
        # Fallback to RMS-based heuristic if split fails
        try:
            hop_length = 256
            frame_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(frame_energy)), sr=sr, hop_length=hop_length)
            energy_norm = (frame_energy - frame_energy.min()) / (frame_energy.ptp() + 1e-9)
            low_mask = energy_norm < 0.35
            breath_idxs = []
            i = 0
            L = len(low_mask)
            while i < L:
                if low_mask[i]:
                    j = i
                    while j + 1 < L and low_mask[j + 1]:
                        j += 1
                    center = (i + j) // 2
                    breath_idxs.append(center)
                    i = j + 1
                else:
                    i += 1
            breath_times = times[breath_idxs].tolist()
            cycles = np.diff(breath_times) if len(breath_times) > 1 else np.array([])
            breath_length_variation = float(np.std(cycles)) if cycles.size else 0.0
            support_score = float(max(0.0, 1.0 - breath_length_variation))
        except Exception:
            breath_times = []
            breath_length_variation = None
            support_score = None

    return {
        "breath_cycles": breath_times,
        "support_score": support_score,
        "breath_length_variation": breath_length_variation,
    }
