"""Training script for the VAD model.

Trains a VADModel on pre-extracted NPZ features from the manifest produced
by ``extract_all.py``.

Usage::

    python ml_new/training/train_vad.py \\
        --manifest ml_new/data/extracted/manifest.csv \\
        --output-dir ml_new/checkpoints/vad \\
        --epochs 30 --batch-size 32 --lr 1e-3

Checkpoints are saved every epoch; the best val-F1 checkpoint is also kept.
A ``train_log.csv`` with per-epoch metrics is written to ``output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.data.feature_dataset import FeatureDataset
from ml_new.models.vad_model import VADModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """Focal binary cross-entropy loss.

    Down-weights easy (confidently correct) predictions, focusing training
    on hard frames.  pos_weight compensates for class imbalance when
    unvoiced frames outnumber voiced frames.

    Args:
        pred: Sigmoid probabilities, same shape as target.
        target: Binary labels (0 or 1), float.
        gamma: Focal exponent (0 = standard BCE, 2 = strong focus on hard examples).
        pos_weight: Weight applied to the positive-class BCE term.

    Returns:
        Scalar mean loss.
    """
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    bce = -(
        pos_weight * target * torch.log(pred)
        + (1 - target) * torch.log(1 - pred)
    )
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for binary classification.

    Args:
        pred_prob: Per-frame voicing probabilities.
        target: Binary ground truth labels.
        threshold: Decision boundary.

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """
    pred = (pred_prob >= threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    manifest: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_len: int = 200,
    focal_gamma: float = 2.0,
    pos_weight: float = 2.0,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    device_str: str = "auto",
) -> None:
    """Run the full training loop and save checkpoints.

    Args:
        manifest: Path to manifest.csv.
        output_dir: Directory for checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        lr: Initial learning rate for AdamW.
        seq_len: Frames per training window.
        focal_gamma: Focal loss exponent.
        pos_weight: Positive-class weight in loss.
        hidden_size: GRU hidden dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout between GRU layers.
        device_str: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    # Datasets
    train_ds = FeatureDataset(manifest, seq_len=seq_len, split="train", task="vad")
    val_ds = FeatureDataset(manifest, seq_len=seq_len, split="val", task="vad")
    log.info("Train: %d clips | Val: %d clips", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = VADModel(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    log.info("VADModel parameters: %d", model.param_count())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    best_f1 = 0.0

    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "epoch", "train_loss", "val_loss",
            "val_accuracy", "val_precision", "val_recall", "val_f1", "lr",
        ])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                hcqt = batch["hcqt"].to(device)
                vad_feats = batch["vad_features"].to(device)
                vad_target = batch["vad"].to(device)

                pred, _ = model(hcqt, vad_feats)
                loss = focal_bce(pred, vad_target, gamma=focal_gamma, pos_weight=pos_weight)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            scheduler.step()

            # --- validate ---
            model.eval()
            val_loss = 0.0
            all_preds: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []

            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch["hcqt"].to(device)
                    vad_feats = batch["vad_features"].to(device)
                    vad_target = batch["vad"].to(device)

                    pred, _ = model(hcqt, vad_feats)
                    val_loss += focal_bce(pred, vad_target, gamma=focal_gamma, pos_weight=pos_weight).item()
                    all_preds.append(pred.cpu())
                    all_targets.append(vad_target.cpu())

            val_loss /= max(len(val_loader), 1)
            metrics = binary_metrics(
                torch.cat(all_preds), torch.cat(all_targets)
            )

            current_lr = scheduler.get_last_lr()[0]
            log.info(
                "Epoch %d/%d | loss=%.4f val_loss=%.4f | acc=%.3f prec=%.3f rec=%.3f f1=%.3f | lr=%.2e",
                epoch, epochs, train_loss, val_loss,
                metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1"], current_lr,
            )

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{metrics['accuracy']:.6f}",
                "val_precision": f"{metrics['precision']:.6f}",
                "val_recall": f"{metrics['recall']:.6f}",
                "val_f1": f"{metrics['f1']:.6f}",
                "lr": f"{current_lr:.2e}",
            }
            writer.writerow(row)
            fh.flush()

            # Save latest checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": metrics["f1"],
                "config": {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
            }
            torch.save(ckpt, output_dir / "latest.pt")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(ckpt, output_dir / "best.pt")
                log.info("  → New best F1=%.4f saved.", best_f1)

    log.info("Training complete. Best val F1=%.4f", best_f1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAD model on extracted NPZ features.")
    p.add_argument("--manifest", type=Path, default=Path("ml_new/data/extracted/manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("ml_new/checkpoints/vad"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        manifest=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device_str=args.device,
    )
