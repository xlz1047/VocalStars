"""PyTorch Dataset for pre-extracted HCQT + VAD feature NPZ files.

Reads the manifest CSV produced by ``extract_all.py`` and serves fixed-length
windows suitable for training VAD and pitch models.

Singer-level splits ensure no singer appears in more than one partition,
preventing the model from memorising singer-specific timbre.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

from ml_new.feature_extraction.breath_labels import derive_breath_labels
from ml_new.feature_extraction.onset_labels import derive_onset_labels


class FeatureDataset(Dataset):
    """Dataset that serves windowed HCQT + VAD features from NPZ files.

    Args:
        manifest_csv: Path to the ``manifest.csv`` produced by
            ``extract_all.py``.
        seq_len: Number of frames per training example (200 ≈ 2 s at 10 ms hop).
        split: Which partition to load: ``"train"``, ``"val"``, or ``"test"``.
        task: Which labels to return.  ``"both"`` returns f0_hz + vad.
            ``"breath"`` returns vad_features + breath labels derived at load time.
            ``"onset"`` returns hcqt first harmonic + onset labels derived from f0_hz.
        val_frac: Fraction of singers reserved for validation.
        test_frac: Fraction of singers reserved for test.
        seed: Random seed for reproducible singer assignment.
        augment: If True, apply random pitch-shift augmentation (train split only).
        shift_semitones: Maximum pitch shift magnitude in semitones.
        bins_per_octave: CQT resolution used to adjust f0_hz labels after shift.
        smooth_f0: If True, apply a 5-frame median filter to f0_hz labels (train only).
            Removes single-frame YIN spikes without affecting true vibrato.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        seq_len: int = 200,
        split: Literal["train", "val", "test"] = "train",
        task: Literal["vad", "pitch", "both", "breath", "onset"] = "both",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        shift_semitones: int = 3,
        bins_per_octave: int = 36,
        smooth_f0: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.task = task
        self.augment = augment and (split == "train")
        self.smooth_f0 = smooth_f0 and (split == "train")
        self.shift_semitones = shift_semitones
        self.bins_per_octave = bins_per_octave

        rows = _load_manifest(manifest_csv)
        rows = _filter_by_split(rows, split, val_frac, test_frac, seed)

        # Only keep clips long enough to sample a full window.
        self._rows = [r for r in rows if int(r["n_frames"]) >= seq_len]
        if not self._rows:
            raise RuntimeError(
                f"No clips with n_frames >= {seq_len} in split='{split}'. "
                "Try a shorter seq_len or check the manifest."
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one windowed training example.

        Args:
            idx: Index into the filtered manifest.

        Returns:
            Dict with tensors:
            - ``hcqt``  (H, n_bins, seq_len) float32
            - ``vad_features``  (3, seq_len) float32
            - ``f0_hz``  (seq_len,) float32  [only if task != "vad"]
            - ``vad``  (seq_len,) float32  [only if task != "pitch"]
        """
        row = self._rows[idx]
        data = np.load(row["npz_path"], allow_pickle=False)

        T = data["hcqt"].shape[2]
        start = random.randint(0, T - self.seq_len)
        sl = slice(start, start + self.seq_len)

        hcqt = data["hcqt"][:, :, sl].copy()  # (H, n_bins, seq_len)
        f0_hz = data["f0_hz"][sl].copy() if self.task in ("pitch", "both") else None

        # Median-filter f0 labels to remove single-frame YIN spikes
        if self.smooth_f0 and f0_hz is not None:
            voiced = f0_hz > 0
            if voiced.any():
                smoothed = median_filter(f0_hz, size=5)
                # Keep voiced mask from original; smoothed value 0 at transitions falls back to original
                f0_hz = np.where(voiced, np.where(smoothed > 0, smoothed, f0_hz), 0.0).astype(np.float32)

        # Pitch-shift augmentation: roll HCQT bins and adjust f0 labels
        if self.augment and random.random() < 0.5:
            k = random.randint(-self.shift_semitones, self.shift_semitones)
            if k != 0:
                n_bins = hcqt.shape[1]
                # Shift bins by k positions
                hcqt = np.roll(hcqt, k, axis=1)
                # Zero-fill the wrapped edge (log-magnitude → silence = log(1e-8))
                silence = np.log(1e-8).astype(np.float32)
                if k > 0:
                    hcqt[:, :k, :] = silence
                else:
                    hcqt[:, k:, :] = silence
                # Adjust f0 labels: shift by k bins = k/bins_per_octave octaves
                if f0_hz is not None:
                    ratio = 2.0 ** (k / self.bins_per_octave)
                    f0_hz = np.where(f0_hz > 0, f0_hz * ratio, 0.0).astype(np.float32)

        out: dict[str, torch.Tensor] = {
            "hcqt": torch.from_numpy(hcqt),
            "vad_features": torch.from_numpy(data["vad_features"][:, sl]),
        }

        if self.task in ("pitch", "both"):
            out["f0_hz"] = torch.from_numpy(f0_hz)
            # voiced_probs: pyin confidence weights (1.0 for yin-extracted clips)
            if "voiced_probs" in data:
                out["voiced_probs"] = torch.from_numpy(data["voiced_probs"][sl].copy())
            else:
                out["voiced_probs"] = torch.ones(self.seq_len)

        if self.task in ("vad", "both"):
            out["vad"] = torch.from_numpy(data["vad"][sl].astype(np.float32))

        if self.task == "breath":
            vad_arr = data["vad"][sl].astype(np.float32)
            vad_feats_arr = data["vad_features"][:, sl]
            out["breath"] = torch.from_numpy(
                derive_breath_labels(vad_arr, vad_feats_arr)
            )

        if self.task == "onset":
            f0_arr = data["f0_hz"][sl].copy()
            out["hcqt_h0"] = torch.from_numpy(hcqt[0])  # (n_bins, seq_len)
            out["onset"] = torch.from_numpy(derive_onset_labels(f0_arr))

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_csv: str | Path) -> list[dict]:
    with open(manifest_csv, newline="") as fh:
        return list(csv.DictReader(fh))


def _filter_by_split(
    rows: list[dict],
    split: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> list[dict]:
    """Assign singers to splits and filter rows accordingly."""
    rng = random.Random(seed)
    all_singers = sorted({r["singer_id"] for r in rows})
    rng.shuffle(all_singers)

    n = len(all_singers)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_singers = set(all_singers[:n_test])
    val_singers = set(all_singers[n_test: n_test + n_val])
    train_singers = set(all_singers[n_test + n_val:])

    target = {"train": train_singers, "val": val_singers, "test": test_singers}[split]
    return [r for r in rows if r["singer_id"] in target]
