"""Fine-tune an acoustic technique classifier on top of a frozen backbone.

Architecture
------------
Frozen: UnifiedVocalModel backbone (harmonic conv + 2-layer GRU)
Trainable: AcousticTechniqueClassifier
  input = concat(clip_repr [128-D], acoustic_features [10-D])
  → Linear(138→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→20)

The 10 acoustic features are derived entirely from pre-extracted NPZ arrays
(RMS, spectral flatness, ZCR statistics + voiced ratio + F0 statistics).
No audio re-read needed during training.

Why this works better than just fine-tuning the head alone (46 %)
------------------------------------------------------------------
The backbone encodes pitch / breath / onset well but the mean-pooled clip
representation carries limited technique-specific signal.  The acoustic
features add direct breathiness (flatness), energy (RMS), and pitch-variation
(F0 std / range) cues that directly separate the 20 technique classes.

Usage::

    python ml_new/training/finetune_technique.py \\
        --base-checkpoint ml_new/checkpoints/unified/best.pt \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --output-dir ml_new/checkpoints/unified_tech \\
        --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.data.unified_dataset import technique_stratified_split, TECHNIQUE_UNKNOWN
from ml_new.models.unified_model import UnifiedVocalModel, N_TECHNIQUES, TECHNIQUE_VOCAB
from ml_new.models.acoustic_technique import (
    AcousticTechniqueClassifier, extract_acoustic_features, N_ACOUSTIC,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AcousticTechniqueDataset(Dataset):
    """Returns one random window + full-clip acoustic features per sample."""

    def __init__(self, rows: list[dict], seq_len: int = 200) -> None:
        from ml_new.models.unified_model import TECHNIQUE_TO_IDX
        self._rows: list[dict] = []
        for r in rows:
            idx = TECHNIQUE_TO_IDX.get(r.get("technique", ""), TECHNIQUE_UNKNOWN)
            if idx >= 0 and int(r["n_frames"]) >= seq_len:
                self._rows.append({**r, "_tech_idx": idx})
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        import random
        row  = self._rows[i]
        data = np.load(row["npz_path"], allow_pickle=False)

        # Random window for HCQT (feeds the backbone)
        T     = data["hcqt"].shape[2]
        start = random.randint(0, T - self.seq_len)
        sl    = slice(start, start + self.seq_len)

        # Acoustic features from the FULL clip (not the random window)
        acoustic = extract_acoustic_features(
            data["vad_features"],
            data["f0_hz"],
            data["vad"],
        )

        return {
            "hcqt":          torch.from_numpy(data["hcqt"][:, :, sl].copy()),
            "vad_features":  torch.from_numpy(data["vad_features"][:, sl]),
            "acoustic_feats": torch.from_numpy(acoustic),
            "technique_idx": torch.tensor(row["_tech_idx"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_weights(rows: list[dict], smoothing: float = 0.5) -> torch.Tensor:
    from ml_new.models.unified_model import TECHNIQUE_TO_IDX
    counts = np.zeros(N_TECHNIQUES, dtype=np.float64) + smoothing
    for r in rows:
        idx = TECHNIQUE_TO_IDX.get(r.get("technique", ""), TECHNIQUE_UNKNOWN)
        if idx >= 0:
            counts[idx] += 1.0
    w = counts.sum() / (N_TECHNIQUES * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def _compute_normalisation(
    train_ds: AcousticTechniqueDataset,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std by loading only the small NPZ arrays.

    Skips HCQT (largest array) to keep this fast — only vad_features, f0_hz,
    and vad are needed for acoustic feature extraction.
    """
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(train_ds), size=min(max_samples, len(train_ds)), replace=False)
    feats_list: list[np.ndarray] = []
    for i in indices:
        row = train_ds._rows[int(i)]
        with np.load(row["npz_path"], allow_pickle=False) as npz:
            acoustic = extract_acoustic_features(
                npz["vad_features"],
                npz["f0_hz"],
                npz["vad"],
            )
        feats_list.append(acoustic)
    feats = np.stack(feats_list)
    mean  = feats.mean(axis=0)
    std   = feats.std(axis=0).clip(min=1e-6)
    return mean, std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    backbone:   UnifiedVocalModel,
    classifier: AcousticTechniqueClassifier,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    *,
    epochs:       int,
    lr:           float,
    weight_decay: float,
    class_weights: torch.Tensor,
    output_dir:   Path,
    device:       torch.device,
    patience:     int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cw = class_weights.to(device)

    optimiser = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_acc   = 0.0
    no_improve = 0

    log_path = output_dir / "finetune_acoustic_log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_acc"])

        for ep in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            classifier.train()
            total_loss, total_n = 0.0, 0
            for batch in train_loader:
                hcqt   = batch["hcqt"].to(device)
                vad_f  = batch["vad_features"].to(device)
                acou   = batch["acoustic_feats"].to(device)
                labels = batch["technique_idx"].to(device)

                with torch.no_grad():
                    clip_repr, _ = backbone.encode_clip(hcqt, vad_f)

                tech_logits = classifier(clip_repr, acou)
                loss = F.cross_entropy(tech_logits, labels, weight=cw)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimiser.step()

                total_loss += loss.item() * len(labels)
                total_n    += len(labels)

            scheduler.step()
            avg_loss = total_loss / max(1, total_n)

            # ── Validate ─────────────────────────────────────────────────
            classifier.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    hcqt   = batch["hcqt"].to(device)
                    vad_f  = batch["vad_features"].to(device)
                    acou   = batch["acoustic_feats"].to(device)
                    labels = batch["technique_idx"].to(device)

                    clip_repr, _ = backbone.encode_clip(hcqt, vad_f)
                    preds = classifier(clip_repr, acou).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total   += len(labels)

            val_acc = correct / max(1, total)
            writer.writerow([ep, f"{avg_loss:.4f}", f"{val_acc:.4f}"])
            fh.flush()

            log.info(
                "ep %02d | train_loss=%.4f | val_acc=%.1f%%",
                ep, avg_loss, val_acc * 100,
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {"classifier_state_dict": classifier.state_dict(),
                     "epoch": ep, "val_tech_acc": val_acc},
                    output_dir / "acoustic_best.pt",
                )
                log.info("  ✓ new best (%.1f%%)", val_acc * 100)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info("Early stop — no improvement for %d epochs.", patience)
                    break

    torch.save(
        {"classifier_state_dict": classifier.state_dict()},
        output_dir / "acoustic_latest.pt",
    )
    log.info("Fine-tune complete. Best val technique accuracy: %.1f%%", best_acc * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune AcousticTechniqueClassifier on frozen backbone."
    )
    p.add_argument("--base-checkpoint", type=Path,
                   default=Path("ml_new/checkpoints/unified/best.pt"))
    p.add_argument("--manifest", type=Path,
                   default=Path("ml_new/data/extracted_pyin/manifest.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("ml_new/checkpoints/unified_tech"))
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seq-len",      type=int,   default=200)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--device",       type=str,   default=None)
    args = p.parse_args(argv)

    if args.device is None:
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    log.info("device: %s", device)

    # ── Load frozen backbone ──────────────────────────────────────────────
    backbone = UnifiedVocalModel().to(device)
    if args.base_checkpoint.exists():
        ckpt = torch.load(str(args.base_checkpoint), map_location=device, weights_only=True)
        backbone.load_state_dict(ckpt.get("model_state_dict", ckpt))
        log.info("Loaded backbone: %s", args.base_checkpoint)
    else:
        log.warning("Checkpoint not found — random init: %s", args.base_checkpoint)
    backbone.eval()
    for p_ in backbone.parameters():
        p_.requires_grad = False

    # ── Splits ───────────────────────────────────────────────────────────
    train_rows, val_rows, _ = technique_stratified_split(args.manifest)
    log.info("Stratified split: train=%d  val=%d", len(train_rows), len(val_rows))

    from collections import Counter
    val_techs = Counter(r.get("technique", "") for r in val_rows)
    log.info("Val classes: %d unique techniques", len(val_techs))

    train_ds = AcousticTechniqueDataset(train_rows, seq_len=args.seq_len)
    val_ds   = AcousticTechniqueDataset(val_rows,   seq_len=args.seq_len)
    log.info("After filtering unknown: train=%d  val=%d", len(train_ds), len(val_ds))

    # ── Normalisation from training split ────────────────────────────────
    log.info("Computing acoustic feature normalisation from training data …")
    feat_mean, feat_std = _compute_normalisation(train_ds)
    log.info("Feature means: %s", np.round(feat_mean, 3))
    log.info("Feature stds : %s", np.round(feat_std,  3))

    # ── Build classifier ─────────────────────────────────────────────────
    classifier = AcousticTechniqueClassifier().to(device)
    classifier.set_normalisation(feat_mean, feat_std)
    log.info("AcousticTechniqueClassifier params: %d", classifier.param_count())

    # ── Data loaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    cw = _class_weights(train_rows)
    log.info("Class weights: min=%.2f  max=%.2f", cw.min(), cw.max())

    _train(
        backbone, classifier,
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=cw,
        output_dir=args.output_dir,
        device=device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
