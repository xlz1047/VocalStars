"""Derive per-frame note onset labels from pre-extracted f0_hz arrays.

No audio re-read is required — operates purely on the f0_hz array stored in
every NPZ file.

An onset occurs at any frame where:
  - The current frame is voiced (f0 > 0), AND
  - Either the previous frame was unvoiced (voiced after silence), OR
  - The pitch jump from the previous voiced frame exceeds a semitone threshold
    (the singer started a new note mid-phrase without a silence gap).
"""

from __future__ import annotations

import numpy as np


def derive_onset_labels(
    f0_hz: np.ndarray,
    semitone_jump_thresh: float = 1.5,
) -> np.ndarray:
    """Compute per-frame binary note onset labels from f0_hz.

    Args:
        f0_hz: ``(T,)`` float32 — fundamental frequency in Hz; 0.0 = unvoiced.
        semitone_jump_thresh: Minimum pitch jump (in semitones) between two
            consecutive voiced frames to be considered a new note onset.
            Vibrato typically stays within 1 semitone.

    Returns:
        ``(T,)`` float32 array with 1.0 at onset frames, 0.0 elsewhere.
    """
    T = len(f0_hz)
    labels = np.zeros(T, dtype=np.float32)
    thresh_cents = semitone_jump_thresh * 100.0

    for t in range(1, T):
        if f0_hz[t] <= 0:
            continue
        if f0_hz[t - 1] <= 0:
            labels[t] = 1.0
        else:
            cents = abs(1200.0 * np.log2(f0_hz[t] / f0_hz[t - 1]))
            if cents > thresh_cents:
                labels[t] = 1.0

    return labels
