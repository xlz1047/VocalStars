"""NUS-48E dataset loader: 48 English song recordings from 12 amateur singers.

Expected directory structure::

    <root>/
        <singer_name>/
            <song_name>/
                <singer_name>_<song_name>.wav
                <singer_name>_<song_name>.csv  # frame-level pitch annotations

Each CSV contains columns: ``time``, ``pitch_hz``.  A value of ``0`` for
``pitch_hz`` indicates an unvoiced frame.  Singer ID is the top-level folder
name (e.g. ``ADIZ``, ``JLEE``).

Implement ``_get_filepaths`` and ``_extract_labels`` once the dataset is
downloaded to ``ml/data/raw/nus48e/``.
"""

import numpy as np

from ml.data.base_dataset import SingingDataset


class NUS48EDataset(SingingDataset):
    """Dataset loader for NUS-48E.

    Loads English song recordings from 12 amateur singers with frame-level
    pitch annotations.

    Args:
        root_dir: Path to the NUS-48E root directory.
        split: Split identifier (``"train"``, ``"val"``, ``"test"``).
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Enumerate all wav/csv pairs and extract singer IDs.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                directory structure is confirmed.
        """
        raise NotImplementedError(
            "NUS48EDataset._get_filepaths: download the dataset to ml/data/raw/nus48e/ first."
        )

    def _extract_labels(self, audio: np.ndarray, sr: int, meta: dict) -> dict:
        """Extract pitch, onset, breath, and singer labels.

        Raises:
            NotImplementedError: Until the dataset is downloaded and the
                annotation CSV format is confirmed.
        """
        raise NotImplementedError(
            "NUS48EDataset._extract_labels: implement after inspecting annotation CSVs."
        )
