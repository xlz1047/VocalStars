"""Abstract base dataset: shared frame-slicing, augmentation, and collation logic."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.feature_extraction.audio_utils import load_audio
from ml.feature_extraction.mel import MelExtractor


class SingingDataset(Dataset, ABC):
    """Abstract base for all singing datasets in VocalStars.

    Subclasses implement ``_get_filepaths`` to enumerate sample dicts and
    ``_extract_labels`` to derive per-sample supervision.  This class owns the
    common load → mel → label pipeline so every dataset inherits identical
    input/output tensors.

    Args:
        root_dir: Absolute path to the dataset root on disk.
        split: Dataset split identifier, e.g. ``"train"``, ``"val"``, ``"test"``.
        sr: Target sample rate in Hz.  Audio is resampled on load if needed.
    """

    def __init__(self, root_dir: str, split: str = "train", sr: int = 16000) -> None:
        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        self._mel = MelExtractor(sr=sr)
        self._files: list[dict] = self._get_filepaths()

    @abstractmethod
    def _get_filepaths(self) -> list[dict]:
        """Return a list of sample metadata dicts.

        Each dict must contain at minimum the key ``"audio_path"`` (str).
        Optional keys: ``"pitch_path"`` (str), ``"singer_id"`` (str).

        Returns:
            List of metadata dicts, one per sample.
        """

    @abstractmethod
    def _extract_labels(self, audio_path: str, meta: dict) -> dict:
        """Derive supervision labels for a single sample.

        Args:
            audio_path: Path to the audio file.
            meta: The metadata dict for this sample (same object returned by
                ``_get_filepaths``), may contain ``"pitch_path"`` etc.

        Returns:
            Dict with keys:
                pitch_hz (float): Median voiced fundamental frequency in Hz.
                onset_frames (np.ndarray): Frame indices of detected onsets.
                breath_bool (bool): True if the clip begins with a breath.
                singer_id (str): Dataset-specific singer identifier.
        """

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Return a single (mel_tensor, labels) pair.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Tuple of:
                mel_tensor: Float32 tensor of shape ``(1, 128, T)`` containing
                    log-mel spectrogram values.
                labels: Dict with keys pitch_hz, onset_frames, breath_bool,
                    singer_id (see ``_extract_labels``).
        """
        meta = self._files[idx]
        audio: np.ndarray = load_audio(meta["audio_path"], sr=self.sr)
        mel_tensor: torch.Tensor = self._mel.compute_tensor(audio)  # (1, 128, T)
        labels: dict = self._extract_labels(meta["audio_path"], meta)
        return mel_tensor, labels
