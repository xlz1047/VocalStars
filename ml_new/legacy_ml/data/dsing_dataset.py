"""DSing dataset loader: solo singing recordings aligned with DALI annotations.

Expected directory structure::

    <root>/
        <song_id>/
            <song_id>.wav
            <song_id>.json     # DALI-format annotation with note-level timestamps
        ...

DALI annotations provide note, word, and phoneme-level alignments in JSON.
Pitch can be derived from the note-level ``f0`` field (Hz).  Singer IDs are
encoded in the annotation metadata under ``"singer"``.

Implement ``_get_filepaths`` and ``_extract_labels`` once the dataset is
downloaded to ``ml/data/raw/dsing/`` and the DALI annotation format is
confirmed against the specific DSing release being used.
"""

import numpy as np

from ml_new.legacy_ml.data.base_dataset import SingingDataset


class DSingDataset(SingingDataset):
    """Dataset loader for DSing.

    Loads solo singing recordings paired with DALI-format JSON annotations for
    note-level pitch and timing supervision.

    Args:
        root_dir: Path to the DSing root directory.
        split: Split identifier (``"train"``, ``"val"``, ``"test"``).
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Enumerate all wav/json pairs.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                annotation format is confirmed.
        """
        raise NotImplementedError(
            "DSingDataset._get_filepaths: download the dataset to ml/data/raw/dsing/ first."
        )

    def _extract_labels(self, audio: np.ndarray, sr: int, meta: dict) -> dict:
        """Extract pitch, onset, breath, and singer labels from DALI annotations.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                DALI JSON structure is confirmed.
        """
        raise NotImplementedError(
            "DSingDataset._extract_labels: implement after inspecting DALI JSON annotations."
        )
