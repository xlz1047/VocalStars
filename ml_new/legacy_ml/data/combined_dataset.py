"""CombinedDataset: merges all individual dataset loaders with configurable per-source weights."""

import bisect
import os

import torch
from torch.utils.data import Dataset

from ml_new.legacy_ml.data.base_dataset import SingingDataset


class CombinedSingingDataset(Dataset):
    """Wraps multiple ``SingingDataset`` instances as a single concatenated dataset.

    Each sample returned by ``__getitem__`` includes a ``"dataset_source"`` key
    in its label dict identifying which constituent dataset produced it.  This
    supports dataset-aware weighted sampling in the training loop.

    Args:
        datasets: List of ``SingingDataset`` instances to concatenate.
    """

    def __init__(self, datasets: list[SingingDataset]) -> None:
        if not datasets:
            raise ValueError("datasets must be a non-empty list")
        self._datasets = datasets
        self._offsets: list[int] = []
        total = 0
        for ds in datasets:
            self._offsets.append(total)
            total += len(ds)
        self._total = total

    def __len__(self) -> int:
        """Return the total number of samples across all constituent datasets."""
        return self._total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Return the (mel_tensor, labels) pair at the given global index.

        The returned ``labels`` dict has all keys produced by the source
        dataset's ``_extract_labels``, plus ``"dataset_source"`` (str).

        Args:
            idx: Global integer index in ``[0, len(self))``.

        Returns:
            Tuple of (mel_tensor, labels) with ``labels["dataset_source"]``
            set to the class name of the originating dataset.
        """
        ds_idx = bisect.bisect_right(self._offsets, idx) - 1
        sample_idx = idx - self._offsets[ds_idx]
        mel_tensor, labels = self._datasets[ds_idx][sample_idx]
        labels = {**labels, "dataset_source": type(self._datasets[ds_idx]).__name__}
        return mel_tensor, labels

    def class_weights(self) -> dict[str, int]:
        """Return per-dataset sample counts for weighted sampling.

        Returns:
            Dict mapping dataset class name to its sample count.  Use these
            counts to construct a ``torch.utils.data.WeightedRandomSampler``.
        """
        return {type(ds).__name__: len(ds) for ds in self._datasets}


def build_dataset(data_dir: str, split: str = "train") -> CombinedSingingDataset:
    """Build a combined dataset from all recognised sub-directories in ``data_dir``.

    Each sub-directory is checked for existence before adding; missing datasets
    are silently skipped.  Raises ``AssertionError`` if no datasets are found.

    Args:
        data_dir: Root directory containing per-dataset sub-directories.
        split: Dataset split to load, e.g. ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A ``CombinedSingingDataset`` concatenating all found datasets.
    """
    from ml_new.legacy_ml.data.vocalset_dataset import VocalSetDataset
    from ml_new.legacy_ml.data.gtsinger_dataset import GTSingerDataset
    from ml_new.legacy_ml.data.popbutfy_dataset import PopBuTFyDataset

    datasets: list[SingingDataset] = []

    vocalset_path = os.path.join(data_dir, "vocalset")
    if os.path.exists(vocalset_path):
        datasets.append(VocalSetDataset(vocalset_path, split))

    gtsinger_path = os.path.join(data_dir, "gtsinger")
    if os.path.exists(gtsinger_path):
        datasets.append(GTSingerDataset(gtsinger_path, split))

    popbutfy_path = os.path.join(data_dir, "popbutfy")
    if os.path.exists(popbutfy_path):
        datasets.append(PopBuTFyDataset(popbutfy_path, split))

    if not datasets:
        raise ValueError(f"No datasets found in {data_dir}")
    print(
        f"Loaded {len(datasets)} datasets: "
        + ", ".join(type(d).__name__ for d in datasets)
    )
    return CombinedSingingDataset(datasets)
