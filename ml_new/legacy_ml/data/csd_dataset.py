"""Children's Song Dataset (CSD) loader: Korean and English children's vocal recordings.

Expected directory structure::

    <root>/
        Korean/
            *.wav
            *.csv          # per-song pitch/note annotations (one row per frame)
        English/
            *.wav
            *.csv

Each CSV contains columns: ``time``, ``pitch_hz``, ``note``.  A ``pitch_hz``
value of ``0`` indicates an unvoiced frame.  Singer IDs are not available in
CSD; use the language subfolder name as a proxy (``"Korean"`` / ``"English"``).

Implement ``_get_filepaths`` and ``_extract_labels`` once the dataset is
downloaded to ``ml/data/raw/csd/``.
"""

import numpy as np

from ml_new.legacy_ml.data.base_dataset import SingingDataset


class CSDDataset(SingingDataset):
    """Dataset loader for the Children's Song Dataset (CSD).

    Loads Korean and English children's vocal recordings with frame-level pitch
    annotations.

    Args:
        root_dir: Path to the CSD root directory (containing ``Korean/`` and
            ``English/`` subdirectories).
        split: Split identifier (``"train"``, ``"val"``, ``"test"``).
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Enumerate all wav files and pair them with their CSV annotations.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                directory structure is confirmed.
        """
        raise NotImplementedError(
            "CSDDataset._get_filepaths: download the dataset to ml/data/raw/csd/ first."
        )

    def _extract_labels(self, audio: np.ndarray, sr: int, meta: dict) -> dict:
        """Extract pitch, onset, breath, and singer labels.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                annotation format is confirmed.
        """
        raise NotImplementedError(
            "CSDDataset._extract_labels: implement after inspecting annotation CSVs."
        )
