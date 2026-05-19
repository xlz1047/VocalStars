"""Mel-spectrogram computation: 16 kHz input, configurable n_mels, hop length, and window."""

import numpy as np
import torch
import torchaudio.transforms as T


class MelExtractor:
    """Computes log-mel spectrograms from raw audio.

    Builds the ``torchaudio.transforms.MelSpectrogram`` transform once on
    construction and reuses it across calls, avoiding repeated graph allocation.

    Args:
        sr: Sample rate of the input audio in Hz.
        n_mels: Number of mel filter banks.
        hop_length: Hop size between STFT frames in samples.
        win_length: STFT window length in samples.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 128,
        hop_length: int = 512,
        win_length: int = 2048,
    ) -> None:
        self._transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Compute a log-mel spectrogram and return it as a numpy array.

        Args:
            audio: 1-D float32 numpy array of audio samples.

        Returns:
            2-D float32 numpy array of shape ``(n_mels, T)`` containing
            log-mel values ``log(mel + 1e-9)``.
        """
        return self.compute_tensor(audio).squeeze(0).numpy()

    def compute_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """Compute a log-mel spectrogram and return it as a PyTorch tensor.

        Args:
            audio: 1-D float32 numpy array of audio samples.

        Returns:
            3-D float32 tensor of shape ``(1, n_mels, T)`` containing
            log-mel values ``log(mel + 1e-9)``.
        """
        waveform = torch.from_numpy(audio).unsqueeze(0)
        mel = self._transform(waveform)
        return torch.log(mel + 1e-9)
