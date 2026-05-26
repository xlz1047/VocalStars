"""Fine-tune the technique head of a pre-trained UnifiedVocalModel.

Problem
-------
The main unified training uses a singer-level val split (male2, male3 from
VocalSet). GTSinger-only techniques (glissando, mixed_voice_and_falsetto,
pharyngeal) dominate training data but are absent from that val split, so the
technique head collapses to ~12 % accuracy despite the other tasks thriving.

Fix
---
1. Load the best unified checkpoint.
2. Freeze all parameters except ``technique_head.*``.
3. Build a *technique-stratified* split — each class gets its own 80/10/10
   division, guaranteeing every technique appears in val.
4. Train the technique head alone for up to 25 epochs with AdamW + weighted CE.
5. Save the fine-tuned checkpoint to ``--output-dir/best.pt``.

Expected outcome: val technique accuracy climbs from ~12 % to 65–80 %.
(90 % is unrealistic cross-singer with 20 classes; 70 %+ gives useful advice.)

Usage::

    python ml_new/training/finetune_technique.py \\
        --base-checkpoint ml_new/checkpoints/unified/best.pt \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --output-dir ml_new/checkpoints/unified_tech \\
        --epochs 25
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

from ml_new.data.unified_dataset import (
    technique_stratified_split,
    TECHNIQUE_UNKNOWN,
)
from ml_new.models.unified_model import UnifiedVocalModel, N_TECHNIQUES, TECHNIQUE_VOCAB

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight dataset — returns only what the technique head needs
# ---------------------------------------------------------------------------

class TechniqueDataset(Dataset):
    """Returns one NPZ window per clip, labelled by technique.

    Only clips with a known technique (technique_idx >= 0) are included.
    """

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
        row = self._rows[i]
        data = np.load(row["npz_path"], allow_pickle=False)
        T    = data["hcqt"].shape[2]
        start = random.randint(0, T - self.seq_len)
        sl    = slice(start, start + self.seq_len)
        return {
            "hcqt":         torch.from_numpy(data["hcqt"][:, :, sl].copy()),
            "vad_features": torch.from_numpy(data["vad_features"][:, sl]),
            "technique_idx": torch.tensor(row["_tech_idx"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def _technique_weights(rows: list[dict], smoothing: float = 0.5) -> torch.Tensor:
    from ml_new.models.unified_model import TECHNIQUE_TO_IDX
    counts = np.zeros(N_TECHNIQUES, dtype=np.float64) + smoothing
    for r in rows:
        idx = TECHNIQUE_TO_IDX.get(r.get("technique", ""), TECHNIQUE_UNKNOWN)
        if idx >= 0:
            counts[idx] += 1.0
    weights = counts.sum() / (N_TECHNIQUES * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    model: UnifiedVocalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    patience: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze everything except the technique head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("technique_head.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable params (technique_head only): %d", trainable)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    cw = class_weights.to(device)

    best_acc  = 0.0
    no_improve = 0

    log_path = output_dir / "finetune_log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_acc"])

        for ep in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            model.train()
            total_loss, total_n = 0.0, 0
            for batch in train_loader:
                hcqt   = batch["hcqt"].to(device)
                vad_f  = batch["vad_features"].to(device)
                labels = batch["technique_idx"].to(device)

                _, _, _, _, tech_logits, _ = model(hcqt, vad_f)
                loss = F.cross_entropy(tech_logits, labels, weight=cw)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

                total_loss += loss.item() * len(labels)
                total_n    += len(labels)

            scheduler.step()
            avg_loss = total_loss / max(1, total_n)

            # ── Validate ─────────────────────────────────────────────────
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    hcqt   = batch["hcqt"].to(device)
                    vad_f  = batch["vad_features"].to(device)
                    labels = batch["technique_idx"].to(device)
                    _, _, _, _, tech_logits, _ = model(hcqt, vad_f)
                    preds = tech_logits.argmax(dim=-1)
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
                    {"model_state_dict": model.state_dict(),
                     "epoch": ep,
                     "val_tech_acc": val_acc},
                    output_dir / "best.pt",
                )
                log.info("  ✓ new best (val_tech_acc=%.1f%%)", val_acc * 100)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info("Early stop — no improvement for %d epochs.", patience)
                    break

    torch.save(
        {"model_state_dict": model.state_dict()},
        output_dir / "latest.pt",
    )
    log.info("Fine-tune complete. Best val technique accuracy: %.1f%%", best_acc * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune the technique head of a trained UnifiedVocalModel."
    )
    p.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("ml_new/checkpoints/unified/best.pt"),
        help="Pre-trained unified checkpoint to load backbone from.",
    )
    p.add_argument(
        "--manifest", type=Path,
        default=Path("ml_new/data/extracted_pyin/manifest.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("ml_new/checkpoints/unified_tech"),
    )
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seq-len",      type=int,   default=200)
    p.add_argument("--patience",     type=int,   default=8,
                   help="Early-stop patience (epochs without improvement).")
    p.add_argument("--device",       type=str,   default=None)
    args = p.parse_args(argv)

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    log.info("device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────
    model = UnifiedVocalModel().to(device)
    if args.base_checkpoint.exists():
        ckpt = torch.load(str(args.base_checkpoint), map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        log.info("Loaded backbone from %s", args.base_checkpoint)
    else:
        log.warning("Checkpoint not found — fine-tuning from random init: %s",
                    args.base_checkpoint)

    # ── Stratified splits ─────────────────────────────────────────────────
    train_rows, val_rows, test_rows = technique_stratified_split(args.manifest)
    log.info(
        "Technique-stratified split: train=%d  val=%d  test=%d",
        len(train_rows), len(val_rows), len(test_rows),
    )

    # Log per-class distribution in val
    from collections import Counter
    from ml_new.models.unified_model import TECHNIQUE_TO_IDX
    val_techs = Counter(r.get("technique", "") for r in val_rows)
    log.info("Val technique distribution: %s",
             {k: v for k, v in sorted(val_techs.items())})

    train_ds = TechniqueDataset(train_rows, seq_len=args.seq_len)
    val_ds   = TechniqueDataset(val_rows,   seq_len=args.seq_len)
    log.info("After filtering unknown: train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    cw = _technique_weights(train_rows)
    log.info("technique weights: min=%.2f  max=%.2f", cw.min(), cw.max())

    _train(
        model, train_loader, val_loader,
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
