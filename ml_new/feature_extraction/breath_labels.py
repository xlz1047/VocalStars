"""Derive per-frame breath labels from pre-extracted NPZ vad features.

No audio re-read is required — operates purely on vad_features and vad arrays
that are already stored in every NPZ file.

Breath frames are unvoiced (vad == 0) with moderate RMS energy and
high zero-crossing rate, characteristic of turbulent airflow noise.
Silence frames have near-zero energy and are NOT labelled as breath.
"""

from __future__ import annotations

import numpy as np


def derive_breath_labels(
    vad: np.ndarray,
    vad_features: np.ndarray,
    rms_low: float = 0.003,
    rms_high: float = 0.10,
    zcr_thresh: float = 0.05,
) -> np.ndarray:
    """Compute per-frame binary breath labels from pre-extracted features.

    Args:
        vad: ``(T,)`` bool/int array — 1 = voiced, 0 = unvoiced.
        vad_features: ``(3, T)`` float32 — [RMS energy, spectral flatness, ZCR].
        rms_low: Minimum RMS energy for a breath frame (filters out silence).
        rms_high: Maximum RMS energy for a breath frame (filters out singing bleed).
        zcr_thresh: Minimum ZCR for a breath frame (turbulent noise is high-ZCR).

    Returns:
        ``(T,)`` float32 array with 1.0 at breath frames, 0.0 elsewhere.
    """
    rms = vad_features[0]
    zcr = vad_features[2]

    unvoiced = (vad == 0)
    has_breath_energy = (rms > rms_low) & (rms < rms_high)
    turbulent = zcr > zcr_thresh

    return (unvoiced & has_breath_energy & turbulent).astype(np.float32)
