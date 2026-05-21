"""Main training script: data loading, optimiser setup, W&B logging, and checkpoint saving."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from ml._model.voice_coach import VoiceCoachModel
from ml.data import (
    CombinedSingingDataset,
    CSDDataset,
    DSingDataset,
    MIR1KDataset,
    NUS48EDataset,
    VocalSetDataset,
)
from ml.data.gtsinger_dataset import GTSingerDataset
from ml.data.popbutfy_dataset import PopBuTFyDataset
from ml.data.base_dataset import SingingDataset
from ml.training.evaluate import evaluate_model
from ml.training.losses import MultiTaskLoss

# ── constants ─────────────────────────────────────────────────────────────────
_WARMUP_STEPS = 500
_N_PITCH_BINS = 360
_PITCH_BIN_HZ: torch.Tensor = 32.7 * (
    2 ** (torch.arange(_N_PITCH_BINS, dtype=torch.float32) * 20.0 / 1200.0)
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_datasets(data_dir: str) -> list[SingingDataset]:
    """Instantiate every available dataset under *data_dir*."""
    root = Path(data_dir)
    candidates: list[tuple[type[SingingDataset], str]] = [
        (VocalSetDataset,  "vocalset"),
        (MIR1KDataset,     "mir1k"),
        (CSDDataset,       "csd"),
        (DSingDataset,     "dsing"),
        (NUS48EDataset,    "nus48e"),
        (GTSingerDataset,  "gtsinger"),
        (PopBuTFyDataset,  "popbutfy"),
    ]
    datasets: list[SingingDataset] = []
    for cls, sub in candidates:
        path = root / sub
        if path.exists():
            try:
                ds = cls(str(path))
                if len(ds) > 0:
                    datasets.append(ds)
                    print(f"  Loaded {cls.__name__}: {len(ds)} samples from {path}")
            except NotImplementedError:
                print(f"  Skipped {cls.__name__}: _get_filepaths not yet implemented")
    if not datasets:
        raise RuntimeError(f"No datasets found under {data_dir}. Check --data-dir.")
    return datasets


def _singer_split(
    dataset: CombinedSingingDataset,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split by singer_id to avoid cross-singer data leakage.

    Collects all unique singer IDs across every sample, shuffles them, then
    assigns the last ``test_split`` fraction to test, the next ``val_split``
    fraction to val, and the rest to train.

    Reads singer IDs from the pre-built ``_files`` metadata list rather than
    loading audio, so this runs in O(n) without any disk I/O.
    """
    singer_to_indices: dict[str, list[int]] = defaultdict(list)
    offset = 0
    for ds in dataset._datasets:
        for local_idx, meta in enumerate(ds._files):
            sid = meta.get("singer_id", "unknown")
            singer_to_indices[sid].append(offset + local_idx)
        offset += len(ds._files)

    singers = sorted(singer_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(singers)

    n = len(singers)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))

    test_singers = set(singers[-n_test:])
    val_singers = set(singers[-(n_test + n_val):-n_test])

    train_idx, val_idx, test_idx = [], [], []
    for sid, idxs in singer_to_indices.items():
        if sid in test_singers:
            test_idx.extend(idxs)
        elif sid in val_singers:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    print(
        f"  Split — train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}"
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def _hz_to_pitch_bins(pitch_hz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a batch of Hz values to 360-dim soft binary pitch targets.

    A Gaussian centred on the nearest bin with sigma=1 bin is applied so that
    adjacent bins carry small positive weight — this matches the CREPE-style
    training target used by PitchHead.

    Args:
        pitch_hz: Float tensor of shape (batch,).
        device: Target device.

    Returns:
        Float tensor of shape (batch, 360) with values in [0, 1].
    """
    bins_hz = _PITCH_BIN_HZ.to(device)  # (360,)
    # cents distance between each sample and each bin
    hz_safe = pitch_hz.clamp(min=1e-6).unsqueeze(1)  # (batch, 1)
    cents = 1200.0 * torch.log2(bins_hz.unsqueeze(0) / hz_safe)  # (batch, 360)
    # 1-bin sigma ≈ 20 cents
    targets = torch.exp(-0.5 * (cents / 20.0) ** 2)
    # Zero out bins for unvoiced frames (pitch_hz == 0)
    voiced = (pitch_hz > 0).float().unsqueeze(1)
    return targets * voiced


def _collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, dict]:
    """Collate a list of (mel_tensor, labels) pairs into a single batch dict.

    Onset frames are converted to a dense per-frame binary mask of the same
    temporal length T as the mel input (before backbone downsampling), so the
    training loop can derive a matching target for the rhythm head output.
    Mel spectrograms are right-padded with zeros to the maximum length in the batch.
    """
    mels, label_list = zip(*batch)
    B = len(mels)
    T = max(m.shape[-1] for m in mels)
    mel_batch = torch.zeros(B, 1, 128, T, dtype=mels[0].dtype)
    for i, m in enumerate(mels):
        mel_batch[i, :, :, : m.shape[-1]] = m

    pitch_hz = torch.tensor(
        [l.get("pitch_hz", 0.0) for l in label_list], dtype=torch.float32
    )

    onset_targets = torch.zeros(B, T, dtype=torch.float32)
    for i, l in enumerate(label_list):
        frames = l.get("onset_frames", [])
        if len(frames) > 0:
            valid = torch.tensor(frames, dtype=torch.long)
            valid = valid[valid < T]
            onset_targets[i, valid] = 1.0

    breath_target = torch.tensor(
        [float(l.get("breath_bool", False)) for l in label_list], dtype=torch.float32
    )

    return mel_batch, {
        "pitch_hz": pitch_hz,
        "onset_targets": onset_targets,
        "breath_target": breath_target,
    }


def _warmup_lambda(step: int) -> float:
    """Linear warmup for the first _WARMUP_STEPS steps, then constant 1.0."""
    if step < _WARMUP_STEPS:
        return (step + 1) / _WARMUP_STEPS
    return 1.0


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: VoiceCoachModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: AdamW,
    warmup_scheduler,
    cosine_scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    device: torch.device,
    global_step: int,
    wandb_run,
) -> tuple[dict[str, float], int]:
    """Run one full pass over *loader*, updating weights and schedulers.

    Returns:
        Tuple of (aggregated loss dict with float means, updated global_step).
    """
    model.train()
    totals: dict[str, float] = defaultdict(float)
    n_batches = 0

    for mel, targets_raw in loader:
        mel = mel.to(device, non_blocking=True)

        pitch_bins = _hz_to_pitch_bins(targets_raw["pitch_hz"], device)

        onset_t = targets_raw["onset_targets"].to(device, non_blocking=True)
        breath_t = targets_raw["breath_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            preds = model(mel)
            # onset_probs shape: (batch, 1, T_out) — downsample onset targets to T_out
            T_out = preds["onset_probs"].shape[-1]
            # Simple strided sampling: pick every k-th frame to match backbone stride
            stride = onset_t.shape[-1] // T_out
            onset_t_ds = onset_t[:, ::stride][:, :T_out]

            losses = criterion(
                {
                    "pitch_logits": preds["pitch_logits"],
                    "onset_probs":  preds["onset_probs"],
                    "breath_prob":  preds["breath_prob"],
                },
                {
                    "pitch_bins":    pitch_bins,
                    "onset_targets": onset_t_ds,
                    "breath_target": breath_t,
                },
            )

        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Warmup overrides cosine during warmup phase
        if global_step < _WARMUP_STEPS:
            for g, lr_scale in zip(
                optimizer.param_groups,
                [_warmup_lambda(global_step)] * len(optimizer.param_groups),
            ):
                g["lr"] = g["initial_lr"] * lr_scale
        else:
            cosine_scheduler.step()

        global_step += 1
        n_batches += 1
        for k, v in losses.items():
            totals[k] += v.item()

        if wandb_run is not None:
            wandb_run.log(
                {f"train/{k}": v.item() for k, v in losses.items()},
                step=global_step,
            )

    means = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return means, global_step


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run the full training loop."""
    parser = argparse.ArgumentParser(description="Train VoiceCoachModel")
    parser.add_argument("--data-dir",       required=True,               help="Path to datasets root")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints/",   help="Directory to save checkpoints")
    parser.add_argument("--epochs",         type=int,   default=50,       help="Number of training epochs")
    parser.add_argument("--batch-size",     type=int,   default=32,       help="Batch size")
    parser.add_argument("--lr",             type=float, default=1e-3,     help="Peak learning rate")
    parser.add_argument("--val-split",      type=float, default=0.1,      help="Fraction of singers for validation")
    parser.add_argument("--test-split",     type=float, default=0.1,      help="Fraction of singers for test")
    parser.add_argument("--wandb",          action="store_true",          help="Enable Weights & Biases logging")
    parser.add_argument("--resume",         default=None,                 help="Path to checkpoint to resume from")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore
        wandb_run = wandb.init(project="vocalstars", config=vars(args))

    # ── data ─────────────────────────────────────────────────────────────────
    print("Loading datasets…")
    raw_datasets = _build_datasets(args.data_dir)
    combined = CombinedSingingDataset(raw_datasets)

    train_set, val_set, _ = _singer_split(
        combined, val_split=args.val_split, test_split=args.test_split
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = VoiceCoachModel().to(device)
    model.summary()

    criterion = MultiTaskLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Store initial LR on each param group for warmup rescaling
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(train_loader), 1), eta_min=1e-6
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_val_loss = math.inf
    global_step = 0

    # ── resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", math.inf)
        print(f"Resumed from epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', '?')}")

    # ── loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            None, cosine_scheduler, scaler, device, global_step, wandb_run,
        )

        val_metrics = evaluate_model(model, val_loader, device)
        val_pitch_loss = val_metrics.get("pitch_loss", float("nan"))

        pitch_acc = val_metrics.get("pitch_rpa", 0.0) * 100.0
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_losses['total_loss']:.4f} | "
            f"val_pitch_loss={val_pitch_loss:.4f} | "
            f"pitch_acc={pitch_acc:.1f}%"
        )

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **{f"val/{k}": v for k, v in val_metrics.items()}})

        # ── checkpoint ────────────────────────────────────────────────────────
        config = vars(args)
        ckpt_payload = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_loss":             val_pitch_loss,
            "config":               config,
        }
        ckpt_path = ckpt_dir / f"{epoch:03d}_{val_pitch_loss:.4f}.pt"
        torch.save(ckpt_payload, ckpt_path)

        if val_pitch_loss < best_val_loss:
            best_val_loss = val_pitch_loss
            best_path = ckpt_dir / "best_model.pt"
            torch.save(ckpt_payload, best_path)
            print(f"  ✓ New best model saved ({best_val_loss:.4f})")

    if wandb_run is not None:
        wandb_run.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
