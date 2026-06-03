"""Dataset loaders package for VocalSet, MIR-1K, CSD, DSing, NUS-48E, GTSinger, PopBuTFy."""

from ml_new.legacy_ml.data.base_dataset import SingingDataset
from ml_new.legacy_ml.data.combined_dataset import CombinedSingingDataset, build_dataset
from ml_new.legacy_ml.data.vocalset_dataset import VocalSetDataset
from ml_new.legacy_ml.data.mir1k_dataset import MIR1KDataset
from ml_new.legacy_ml.data.csd_dataset import CSDDataset
from ml_new.legacy_ml.data.dsing_dataset import DSingDataset
from ml_new.legacy_ml.data.nus48e_dataset import NUS48EDataset
from ml_new.legacy_ml.data.gtsinger_dataset import GTSingerDataset
from ml_new.legacy_ml.data.popbutfy_dataset import PopBuTFyDataset

__all__ = [
    "SingingDataset",
    "CombinedSingingDataset",
    "build_dataset",
    "VocalSetDataset",
    "MIR1KDataset",
    "CSDDataset",
    "DSingDataset",
    "NUS48EDataset",
    "GTSingerDataset",
    "PopBuTFyDataset",
]
