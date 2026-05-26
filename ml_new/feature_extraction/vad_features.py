"""Handcrafted VAD (Voice Activity Detection) feature extractor.

Three complementary per-frame features:
  - RMS energy: voiced speech is significantly louder than silence/noise.
  - Spectral flatness (Wiener entropy): voiced speech has a tonal harmonic
    structure (low flatness ≈ 0–0.1); noise/unvoiced is more spectrally flat.
  - Zero crossing rate: voiced speech has a lower ZCR than fricatives/noise
    because voiced frames contain low-frequency periodic energy.

These features are lightweight and can all be computed causally frame-by-frame
at inference time using a fixed-size ring buffer.
"""

from __future__ import annotations

import numpy as np
import librosa


class VADFeatureExtractor:
    """Extracts per-frame voiced/unvoiced discriminant features.

    Args:
        sr: Sample rate in Hz.
        hop_length: Frame shift in samples (160 = 10 ms at 16 kHz).
        frame_length: Analysis window in samples (400 = 25 ms at 16 kHz).
        n_fft: FFT size used for spectral flatness computation.
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,
        frame_length: int = 400,
        n_fft: int = 512,
    ) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.n_fft = n_fft

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Compute the three-channel VAD feature matrix.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Float32 array of shape ``(3, T)`` where the three channels are
            ``[rms, spectral_flatness, zcr]`` in that order.
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        T = min(len(rms), len(flatness), len(zcr))
        return np.stack(
            [rms[:T], flatness[:T], zcr[:T]], axis=0
        ).astype(np.float32)
