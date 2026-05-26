"""Dataset for the unified multi-task vocal model.

Returns all five label types in a single sample:
  - hcqt          (H, n_bins, seq_len)
  - vad_features  (3, seq_len)
  - f0_hz         (seq_len,)   — ground-truth F0 in Hz; 0 = unvoiced
  - voiced_probs  (seq_len,)   — pyin confidence weights
  - vad           (seq_len,)   — binary voiced label
  - breath        (seq_len,)   — derived breath label
  - onset         (seq_len,)   — derived onset label
  - technique_idx  int         — canonical technique class index (clip-level)
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
from ml_new.models.unified_model import TECHNIQUE_TO_IDX, N_TECHNIQUES

# Clips with unknown technique are mapped to this sentinel so they are
# excluded from the technique loss (masked out in the training loop).
TECHNIQUE_UNKNOWN = -1


class UnifiedDataset(Dataset):
    """Windowed multi-task dataset from pre-extracted NPZ files.

    Args:
        manifest_csv: Path produced by ``extract_all.py``.
        seq_len: Frames per training window (200 ≈ 2 s at 10 ms hop).
        split: ``"train"``, ``"val"``, or ``"test"``.
        val_frac: Singer fraction for validation.
        test_frac: Singer fraction for test.
        seed: Reproducible singer shuffle seed.
        augment: Pitch-shift augmentation (train only).
        shift_semitones: Max pitch shift magnitude.
        bins_per_octave: CQT resolution for label adjustment.
        smooth_f0: Median-filter F0 labels on train split.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        seq_len: int = 200,
        split: Literal["train", "val", "test"] = "train",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        shift_semitones: int = 3,
        bins_per_octave: int = 36,
        smooth_f0: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.augment = augment and (split == "train")
        self.smooth_f0 = smooth_f0 and (split == "train")
        self.shift_semitones = shift_semitones
        self.bins_per_octave = bins_per_octave

        rows = _load_manifest(manifest_csv)
        rows = _filter_by_split(rows, split, val_frac, test_frac, seed)
        self._rows = [r for r in rows if int(r["n_frames"]) >= seq_len]
        if not self._rows:
            raise RuntimeError(
                f"No clips with n_frames >= {seq_len} in split='{split}'."
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        row = self._rows[idx]
        data = np.load(row["npz_path"], allow_pickle=False)

        T = data["hcqt"].shape[2]
        start = random.randint(0, T - self.seq_len)
        sl = slice(start, start + self.seq_len)

        hcqt = data["hcqt"][:, :, sl].copy()               # (H, n_bins, seq_len)
        f0_hz = data["f0_hz"][sl].copy().astype(np.float32)
        vad = data["vad"][sl].astype(np.float32)
        vad_features = data["vad_features"][:, sl]          # (3, seq_len)

        # Smooth F0 labels on train to remove single-frame YIN spikes
        if self.smooth_f0:
            voiced = f0_hz > 0
            if voiced.any():
                smoothed = median_filter(f0_hz, size=5)
                f0_hz = np.where(
                    voiced, np.where(smoothed > 0, smoothed, f0_hz), 0.0
                ).astype(np.float32)

        # Pitch-shift augmentation (roll HCQT bins, scale f0 labels)
        if self.augment and random.random() < 0.5:
            k = random.randint(-self.shift_semitones, self.shift_semitones)
            if k != 0:
                hcqt = np.roll(hcqt, k, axis=1)
                silence = np.log(1e-8).astype(np.float32)
                if k > 0:
                    hcqt[:, :k, :] = silence
                else:
                    hcqt[:, k:, :] = silence
                ratio = 2.0 ** (k / self.bins_per_octave)
                f0_hz = np.where(f0_hz > 0, f0_hz * ratio, 0.0).astype(np.float32)

        # pyin confidence weights (1.0 for yin-extracted clips)
        voiced_probs = (
            data["voiced_probs"][sl].copy()
            if "voiced_probs" in data
            else np.ones(self.seq_len, dtype=np.float32)
        )

        # Derived labels (computed from already-loaded arrays — no I/O)
        breath = derive_breath_labels(vad, vad_features)    # (seq_len,)
        onset = derive_onset_labels(f0_hz)                  # (seq_len,)

        # Technique: map manifest string to canonical index
        tech_str = row.get("technique", "")
        technique_idx = TECHNIQUE_TO_IDX.get(tech_str, TECHNIQUE_UNKNOWN)

        return {
            "hcqt":          torch.from_numpy(hcqt),
            "vad_features":  torch.from_numpy(vad_features),
            "f0_hz":         torch.from_numpy(f0_hz),
            "voiced_probs":  torch.from_numpy(voiced_probs),
            "vad":           torch.from_numpy(vad),
            "breath":        torch.from_numpy(breath),
            "onset":         torch.from_numpy(onset),
            "technique_idx": torch.tensor(technique_idx, dtype=torch.long),
        }


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


def technique_stratified_split(
    manifest_csv: str | Path,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split clips stratified by technique label (not singer).

    Each technique class contributes proportionally to every split, so the
    validation and test sets are guaranteed to contain examples of every class
    that appears in training — fixing the problem where singer-level splits
    leave GTSinger-only techniques unseen at validation.

    Args:
        manifest_csv: Path to the manifest produced by ``extract_all.py``.
        val_frac: Fraction of each class's clips to reserve for validation.
        test_frac: Fraction of each class's clips to reserve for test.
        seed: Random seed for reproducible shuffling.

    Returns:
        ``(train_rows, val_rows, test_rows)`` — three lists of manifest dicts.
    """
    rows = _load_manifest(manifest_csv)
    # Group by technique
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("technique", "")].append(r)

    rng = random.Random(seed)
    train_rows, val_rows, test_rows = [], [], []
    for tech, clips in groups.items():
        clips = list(clips)
        rng.shuffle(clips)
        n = len(clips)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))
        test_rows  += clips[:n_test]
        val_rows   += clips[n_test: n_test + n_val]
        train_rows += clips[n_test + n_val:]

    return train_rows, val_rows, test_rows


def technique_class_weights(
    manifest_csv: str | Path,
    split: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for the technique CE loss.

    Args:
        smoothing: Add this count to every class before computing weights
            (Laplace smoothing prevents infinite weights for unseen classes).

    Returns:
        ``(N_TECHNIQUES,)`` float32 weight tensor.
    """
    rows = _load_manifest(manifest_csv)
    rows = _filter_by_split(rows, split, val_frac, test_frac, seed)
    counts = np.zeros(N_TECHNIQUES, dtype=np.float64) + smoothing
    for r in rows:
        idx = TECHNIQUE_TO_IDX.get(r.get("technique", ""), -1)
        if idx >= 0:
            counts[idx] += 1.0
    weights = counts.sum() / (N_TECHNIQUES * counts)
    weights = weights / weights.mean()  # normalise so mean weight = 1
    return torch.tensor(weights, dtype=torch.float32)
