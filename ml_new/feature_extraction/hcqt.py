"""Harmonic Constant-Q Transform (HCQT) feature extractor for singing pitch detection.

HCQT stacks multiple CQT representations, one per harmonic. Each harmonic h uses
fmin*h as its base frequency. Because all harmonics share the same n_bins and
bins_per_octave, the frequency axis is aligned: bin b of harmonic 1 and bin b
of harmonic 2 differ by exactly one octave. Harmonic overtones of a sung pitch
therefore appear as vertical stripes at the same bin position across channels,
making it straightforward for a model to detect the fundamental.

Reference: Bitteur et al. (2017), Salamon et al. (2017), Kim et al. (2019).
"""

from __future__ import annotations

import numpy as np
import librosa


class HCQTExtractor:
    """Extracts Harmonic CQT features from a mono audio array.

    Args:
        sr: Sample rate in Hz.
        fmin: Fundamental frequency of the lowest CQT bin (harmonic 1), in Hz.
            C1 (32.7 Hz) covers the full singing range B0–B6.
        n_harmonics: Number of harmonic layers to stack.
        n_bins: Number of CQT frequency bins per harmonic. With
            bins_per_octave=12 and n_bins=72, this covers 6 octaves.
        bins_per_octave: Frequency resolution. 12 = 1 bin per semitone.
        hop_length: Frame shift in samples (160 = 10 ms at 16 kHz).
    """

    def __init__(
        self,
        sr: int = 16000,
        fmin: float = 32.7,
        n_harmonics: int = 6,
        n_bins: int = 60,
        bins_per_octave: int = 12,
        hop_length: int = 160,
    ) -> None:
        # n_bins=60 (5 octaves at 12 bins/octave) keeps every harmonic's top
        # bin well below the Nyquist limit of 8000 Hz at 16 kHz.
        # H1 covers 32.7–987 Hz (C1–B5), sufficient for the full vocal range.
        self.sr = sr
        self.fmin = fmin
        self.n_harmonics = n_harmonics
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Compute the HCQT of a mono audio signal.

        Args:
            audio: 1-D float32 array of audio samples at ``self.sr``.

        Returns:
            Float32 array of shape ``(n_harmonics, n_bins, T)`` containing
            log-magnitude CQT values.  Unvoiced silence regions will have
            values near ``log(1e-8) ≈ -18.4``.
        """
        layers: list[np.ndarray] = []
        for h in range(1, self.n_harmonics + 1):
            cqt = librosa.cqt(
                audio,
                sr=self.sr,
                hop_length=self.hop_length,
                fmin=self.fmin * h,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
            log_mag = np.log(np.abs(cqt).astype(np.float32) + 1e-8)
            layers.append(log_mag)

        # Different harmonics may produce slightly different T due to varying
        # window lengths; trim all to the shortest before stacking.
        T_min = min(layer.shape[1] for layer in layers)
        layers = [layer[:, :T_min] for layer in layers]

        return np.stack(layers, axis=0)  # (n_harmonics, n_bins, T)

    @property
    def output_shape(self) -> tuple[int, int, str]:
        """Symbolic output shape: (n_harmonics, n_bins, T)."""
        return (self.n_harmonics, self.n_bins, "T")
