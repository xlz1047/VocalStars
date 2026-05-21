"""Synthetic pitch evaluation for `ml.pitch_detection.extract_pitch_features`.

Generates a concatenation of pure sine notes at known frequencies, runs
librosa.pyin to estimate F0, and reports RMSE and voiced detection rates.

Run with: `python -m tests.evaluate_pitch` or via the provided runner.
"""
from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import List

import numpy as np
import librosa
import soundfile as sf


def synth_notes(frequencies: List[float], durations: List[float], sr: int = 22050, amplitude: float = 0.6):
    assert len(frequencies) == len(durations)
    parts = []
    for f, d in zip(frequencies, durations):
        t = np.linspace(0, d, int(sr * d), endpoint=False)
        parts.append(amplitude * np.sin(2 * np.pi * f * t))
    return np.concatenate(parts)


def build_ground_truth_times(frequencies: List[float], durations: List[float], times: np.ndarray):
    # Map each timestamp to the corresponding frequency (or 0 for silence)
    edges = np.concatenate(([0.0], np.cumsum(durations)))
    gt = np.zeros_like(times)
    for i in range(len(frequencies)):
        mask = (times >= edges[i]) & (times < edges[i + 1])
        gt[mask] = frequencies[i]
    return gt


def evaluate_pitch_on_synth(save_path: Path | str | None = None):
    sr = 22050
    # simple melody: A3 (220), A4 (440), E4 (330), silence
    freqs = [220.0, 440.0, 329.63, 0.0]
    durs = [1.5, 1.5, 1.5, 1.0]

    y = synth_notes(freqs, durs, sr=sr)
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_path = tmp.name
        tmp.close()

    sf.write(save_path, y, sr)

    # Run librosa.pyin just like the pipeline does
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    times = librosa.times_like(f0, sr=sr)

    # Ground truth per frame
    gt = build_ground_truth_times(freqs, durs, times)

    # Compare only where ground truth is voiced (gt>0)
    voiced_gt_mask = gt > 0
    est = np.nan_to_num(f0, nan=0.0)

    # RMSE on voiced frames
    if np.any(voiced_gt_mask):
        rmse = float(np.sqrt(np.mean((est[voiced_gt_mask] - gt[voiced_gt_mask]) ** 2)))
    else:
        rmse = None

    # Voiced detection: true positive rate and false positive rate
    detected_voiced = est > 0
    tp = int(np.sum(detected_voiced & voiced_gt_mask))
    fn = int(np.sum(~detected_voiced & voiced_gt_mask))
    fp = int(np.sum(detected_voiced & ~voiced_gt_mask))
    tn = int(np.sum(~detected_voiced & ~voiced_gt_mask))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    results = {
        "rmse_hz": rmse,
        "voiced_true_positive_rate": tpr,
        "voiced_false_positive_rate": fpr,
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "save_path": str(save_path),
    }

    # Note-level evaluation: compare median estimated pitch inside each ground-truth note
    edges = np.concatenate(([0.0], np.cumsum(durs)))
    note_results = []
    note_errors = []
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        mask = (times >= edges[i]) & (times < edges[i + 1])
        if not np.any(mask):
            note_results.append({"target_hz": f, "detected": False})
            continue
        est_median = float(np.median(est[mask])) if np.any(est[mask] > 0) else 0.0
        detected = est_median > 0
        err_hz = abs(est_median - f) if detected else None
        # cents error (approx): 1200*log2(est/target)
        err_cents = float(1200 * np.log2(est_median / f)) if detected and est_median > 0 else None
        if err_hz is not None:
            note_errors.append(err_hz)
        note_results.append({"target_hz": f, "estimated_median_hz": est_median, "error_hz": err_hz, "error_cents": err_cents, "detected": detected})

    note_rmse = float(np.sqrt(np.mean(np.array(note_errors) ** 2))) if len(note_errors) else None
    results["note_level"] = {"notes": note_results, "note_rmse_hz": note_rmse}

    return results


def run():
    print("Running synthetic pitch evaluation...")
    res = evaluate_pitch_on_synth()
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
