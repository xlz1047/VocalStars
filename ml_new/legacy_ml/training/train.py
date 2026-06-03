"""Main training script: data loading, optimiser setup, W&B logging, and checkpoint saving."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
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
from ml.pitch_detection.model import PitchHead
from ml.training.evaluate import evaluate_model
from ml.training.losses import MultiTaskLoss
from ml.training.npz_dataset import NanoPitchDataset, f0_to_posteriorgram

# ── constants ─────────────────────────────────────────────────────────────────
_WARMUP_STEPS = 500
_PITCH_BIN_HZ: torch.Tensor = PitchHead.FMIN * (
    2 ** (torch.arange(PitchHead.N_BINS, dtype=torch.float32) * PitchHead.CENTS_PER_BIN / 1200.0)
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
    """Split by singer_id to avoid cross-singer data leakage."""
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


def _hz_to_pitch_bins(
    pitch_hz: torch.Tensor, T: int, device: torch.device
) -> torch.Tensor:
    """Convert clip-level Hz values to per-frame ``(B, T, 360)`` soft targets.

    The Gaussian is centred on the nearest bin (sigma = 1 bin = 20 cents).
    Unvoiced clips (pitch_hz == 0) produce all-zero targets.  The same clip-level
    pitch is broadcast across all T frames, which is a valid simplification for
    short clips with a single dominant pitch.

    Args:
        pitch_hz: Float tensor of shape ``(B,)``.
        T: Number of frames to broadcast across.
        device: Target device.

    Returns:
        Float tensor of shape ``(B, T, 360)`` with values in ``[0, 1]``.
    """
    bins_hz = _PITCH_BIN_HZ.to(device)                        # (360,)
    hz_safe = pitch_hz.clamp(min=1e-6).unsqueeze(1)           # (B, 1)
    cents = 1200.0 * torch.log2(bins_hz.unsqueeze(0) / hz_safe)  # (B, 360)
    targets_clip = torch.exp(-0.5 * (cents / 20.0) ** 2)      # (B, 360)
    voiced = (pitch_hz > 0).float().unsqueeze(1)               # (B, 1)
    targets_clip = targets_clip * voiced                        # (B, 360)
    return targets_clip.unsqueeze(1).expand(-1, T, -1)         # (B, T, 360)


def _collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, dict]:
    """Collate ``(mel_tensor, labels)`` pairs into a time-major batch.

    Mel spectrograms are zero-padded on the time axis to the maximum length and
    transposed to time-major format ``(B, T, n_mels)`` for the GRU backbone.
    Pitch targets are broadcast from clip-level Hz to per-frame ``(B, T, 360)``
    soft Gaussian labels.

    Args:
        batch: List of ``(mel_tensor, labels)`` pairs.  ``mel_tensor`` has shape
            ``(1, n_mels, T)`` from ``MelExtractor.compute_tensor``.

    Returns:
        Tuple of ``(mel_batch, targets)`` where:
            mel_batch: ``(B, T_max, n_mels)`` float32.
            targets:   Dict with keys ``pitch_hz``, ``onset_targets``,
                       ``breath_target``, ``pitch_bins``.
    """
    mels, label_list = zip(*batch)
    B = len(mels)
    n_mels = mels[0].shape[1]
    T = max(m.shape[-1] for m in mels)

    # Build (B, T, n_mels) time-major batch
    mel_batch = torch.zeros(B, T, n_mels, dtype=mels[0].dtype)
    for i, m in enumerate(mels):
        t_i = m.shape[-1]
        # m: (1, n_mels, T_i) → squeeze → (n_mels, T_i) → transpose → (T_i, n_mels)
        mel_batch[i, :t_i, :] = m.squeeze(0).T

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
        "pitch_hz":      pitch_hz,
        "onset_targets": onset_targets,
        "breath_target": breath_target,
    }


def _warmup_lambda(step: int) -> float:
    """Linear warmup for the first _WARMUP_STEPS steps, then constant 1.0."""
    if step < _WARMUP_STEPS:
        return (step + 1) / _WARMUP_STEPS
    return 1.0


def _augment_mel_batch(
    mel_clean: torch.Tensor,
    mel_noise: torch.Tensor,
    snr_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Mix clean and noise log-mel at a random SNR.

    Args:
        mel_clean: ``(B, T, 40)`` clean log-mel.
        mel_noise: ``(B, T, 40)`` noise log-mel.
        snr_range: ``(min_snr_db, max_snr_db)`` tuple.
        device: Torch device.

    Returns:
        ``(B, T, 40)`` mixed log-mel.
    """
    B = mel_clean.size(0)
    snr_db = (
        torch.rand(B, 1, 1, device=device)
        * (snr_range[1] - snr_range[0])
        + snr_range[0]
    )
    gain_offset = -snr_db * (np.log(10.0) / 20.0)
    return torch.logaddexp(mel_clean, mel_noise + gain_offset)


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: VoiceCoachModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: AdamW,
    cosine_scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    device: torch.device,
    global_step: int,
    wandb_run,
    npz_loader: DataLoader | None = None,
    snr_range: tuple[float, float] = (-5.0, 20.0),
) -> tuple[dict[str, float], int]:
    """Run one full pass over *loader*, updating weights and schedulers.

    If *npz_loader* is provided, each singing-dataset batch is followed by one
    NanoPitch-npz batch (with per-frame pitch and VAD supervision).

    Returns:
        Tuple of (aggregated loss dict with float means, updated global_step).
    """
    model.train()
    totals: dict[str, float] = defaultdict(float)
    n_batches = 0

    npz_iter = iter(npz_loader) if npz_loader is not None else None

    for mel, targets_raw in loader:
        mel = mel.to(device, non_blocking=True)          # (B, T, 40)
        T = mel.shape[1]

        pitch_hz   = targets_raw["pitch_hz"].to(device, non_blocking=True)
        pitch_bins = _hz_to_pitch_bins(pitch_hz, T, device)                # (B, T, 360)
        onset_t    = targets_raw["onset_targets"].to(device, non_blocking=True)  # (B, T)
        breath_t   = targets_raw["breath_target"].to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=device.type == "cuda"):
            preds = model(mel)
            T_out = preds["onset_probs"].shape[-1]
            # Align onset targets to model output length (no-op when T_out == T)
            stride = max(1, T // T_out)
            onset_t_aligned = onset_t[:, ::stride][:, :T_out]

            losses = criterion(
                {
                    "pitch_logits": preds["pitch_logits"],
                    "onset_probs":  preds["onset_probs"],
                    "breath_prob":  preds["breath_prob"],
                    "vad_logits":   preds["vad_logits"],
                },
                {
                    "pitch_bins":    pitch_bins,
                    "onset_targets": onset_t_aligned.unsqueeze(1),
                    "breath_target": breath_t,
                },
            )

        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        _step_schedulers(optimizer, cosine_scheduler, global_step)
        global_step += 1
        n_batches += 1
        for k, v in losses.items():
            totals[k] += v.item()
        if wandb_run is not None:
            wandb_run.log({f"train/{k}": v.item() for k, v in losses.items()}, step=global_step)

        # ── optional NanoPitch npz batch ──────────────────────────────────────
        if npz_iter is not None:
            global_step, n_batches, totals = _train_npz_step(
                model, criterion, optimizer, cosine_scheduler, scaler,
                device, npz_iter, npz_loader, snr_range,
                global_step, n_batches, totals, wandb_run,
            )

    means = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return means, global_step


def _step_schedulers(optimizer, cosine_scheduler, global_step: int) -> None:
    if global_step < _WARMUP_STEPS:
        scale = _warmup_lambda(global_step)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale
    else:
        cosine_scheduler.step()


def _train_npz_step(
    model, criterion, optimizer, cosine_scheduler, scaler,
    device, npz_iter, npz_loader, snr_range,
    global_step, n_batches, totals, wandb_run,
):
    """Process one batch from the NanoPitch npz loader."""
    try:
        mel_clean, mel_noise, vad_gt, f0_gt = next(npz_iter)
    except StopIteration:
        npz_iter = iter(npz_loader)
        mel_clean, mel_noise, vad_gt, f0_gt = next(npz_iter)

    mel_clean = mel_clean.to(device)   # (B, T, 40)
    mel_noise = mel_noise.to(device)
    vad_gt    = vad_gt.to(device)      # (B, T)
    f0_gt     = f0_gt.to(device)       # (B, T)

    # Noise mixing
    mel_mixed = _augment_mel_batch(mel_clean, mel_noise, snr_range, device)

    B, T, _ = mel_mixed.shape
    # Per-frame pitch posteriorgram targets
    f0_np = f0_gt.cpu().numpy()
    pitch_bins_np = np.stack([f0_to_posteriorgram(f0_np[b], n_frames=T) for b in range(B)])
    pitch_bins = torch.from_numpy(pitch_bins_np).to(device)   # (B, T, 360)
    vad_target = vad_gt.unsqueeze(-1)                          # (B, T, 1)

    optimizer.zero_grad(set_to_none=True)
    with autocast(device.type):
        preds = model(mel_mixed)
        T_out = preds["onset_probs"].shape[-1]
        dummy_onset = torch.zeros(B, 1, T_out, device=device)
        dummy_breath = torch.zeros(B, device=device)

        losses = criterion(
            {
                "pitch_logits": preds["pitch_logits"],
                "onset_probs":  dummy_onset,
                "breath_prob":  preds["breath_prob"],
                "vad_logits":   preds["vad_logits"],
            },
            {
                "pitch_bins":    pitch_bins,
                "onset_targets": dummy_onset,
                "breath_target": dummy_breath,
                "vad_target":    vad_target,
            },
        )

    scaler.scale(losses["total_loss"]).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    _step_schedulers(optimizer, cosine_scheduler, global_step)
    global_step += 1
    n_batches += 1
    for k, v in losses.items():
        totals[k] += v.item()
    if wandb_run is not None:
        wandb_run.log({f"train_npz/{k}": v.item() for k, v in losses.items()}, step=global_step)

    return global_step, n_batches, totals


# ── evaluation display ────────────────────────────────────────────────────────

def _print_eval_table(
    epoch: int,
    train_losses: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    """Print a formatted evaluation table for the given epoch."""
    rows = [
        ("Train loss",     f"{train_losses.get('total_loss', float('nan')):.4f}", ""),
        ("Pitch RPA",      f"{val_metrics.get('pitch_rpa', 0.0) * 100:.1f}%",    "±50 cents"),
        ("Pitch RCA",      f"{val_metrics.get('pitch_rca', 0.0) * 100:.1f}%",    "octave-invariant"),
        ("Rhythm F1",      f"{val_metrics.get('onset_f1', 0.0) * 100:.1f}%",     "onset@0.5"),
        ("Breath acc",     f"{val_metrics.get('breath_acc', 0.0) * 100:.1f}%",   "binary"),
        ("VAD acc",        f"{val_metrics.get('vad_acc', 0.0) * 100:.1f}%",      "clip-level proxy"),
        ("Overall score",  f"{val_metrics.get('overall', 0.0) * 100:.1f}%",      "weighted mean"),
        ("Val pitch loss", f"{val_metrics.get('pitch_loss', float('nan')):.4f}",  ""),
    ]
    header = f"  Epoch {epoch:03d}  Detailed Evaluation"
    print(f"┌{'─' * 52}┐")
    print(f"│{header:<52}│")
    print(f"├{'─' * 18}┬{'─' * 11}┬{'─' * 20}┤")
    print(f"│  {'Metric':<16}│  {'Value':<9}│  {'Notes':<18}│")
    print(f"├{'─' * 18}┼{'─' * 11}┼{'─' * 20}┤")
    for label, value, note in rows:
        print(f"│  {label:<16}│  {value:<9}│  {note:<18}│")
    print(f"└{'─' * 18}┴{'─' * 11}┴{'─' * 20}┘")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run the full training loop."""
    parser = argparse.ArgumentParser(description="Train VoiceCoachModel")
    parser.add_argument("--data-dir",              required=True,
                        help="Path to datasets root")
    parser.add_argument("--checkpoint-dir",        default="ml/checkpoints/",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs",                type=int,   default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size",            type=int,   default=32,
                        help="Batch size")
    parser.add_argument("--lr",                    type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--val-split",             type=float, default=0.1,
                        help="Fraction of singers for validation")
    parser.add_argument("--test-split",            type=float, default=0.1,
                        help="Fraction of singers for test")
    parser.add_argument("--wandb",                 action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--resume",                default=None,
                        help="Path to VoiceCoach checkpoint to resume from")
    # NanoPitch integration
    parser.add_argument("--use-pretrained",        action="store_true",
                        help="Initialise backbone and pitch head from NanoPitch checkpoint")
    parser.add_argument("--nanopitch-checkpoint",
                        default=None,
                        help="Path to NanoPitch .pth checkpoint for weight transfer")
    parser.add_argument("--npz-dir",               default=None,
                        help="Directory with NanoPitch clean.npz/noise.npz for co-training")
    parser.add_argument("--npz-seq-len",           type=int,   default=200,
                        help="Frame window length for NanoPitch npz dataset")
    parser.add_argument("--snr-min",               type=float, default=-5.0,
                        help="Min SNR in dB for npz noise mixing")
    parser.add_argument("--snr-max",               type=float, default=20.0,
                        help="Max SNR in dB for npz noise mixing")
    parser.add_argument("--reset-lr",              action="store_true",
                        help="After loading a --resume checkpoint, reset LR to --lr "
                             "instead of continuing from the saved optimizer state")
    parser.add_argument("--freeze-pretrained-epochs", type=int, default=0,
                        help="When using --use-pretrained, freeze encoder and pitch head "
                             "for this many epochs so new heads stabilise first")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    n_workers = min(4, os.cpu_count() or 1)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=_collate_fn,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=_collate_fn,
        pin_memory=pin,
    )

    # ── optional NanoPitch npz co-training data ───────────────────────────────
    npz_loader: DataLoader | None = None
    if args.npz_dir:
        print(f"Loading NanoPitch npz data from {args.npz_dir}…")
        npz_ds = NanoPitchDataset(args.npz_dir, seq_len=args.npz_seq_len)
        npz_loader = DataLoader(
            npz_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin,
        )
        print(f"  NanoPitch dataset: {len(npz_ds)} sequences")

    # ── model ─────────────────────────────────────────────────────────────────
    if args.use_pretrained:
        if not args.nanopitch_checkpoint:
            raise ValueError("--use-pretrained requires --nanopitch-checkpoint <path>")
        print(f"Loading pretrained NanoPitch weights from {args.nanopitch_checkpoint}…")
        model = VoiceCoachModel.from_nanopitch_checkpoint(args.nanopitch_checkpoint)
    else:
        model = VoiceCoachModel()
    model = model.to(device)
    model.summary()

    criterion = MultiTaskLoss()
    if args.use_pretrained:
        pretrained_params = (
            list(model.encoder.parameters()) + list(model.pitch_head.parameters())
        )
        new_params = (
            list(model.rhythm_head.parameters()) + list(model.breath_head.parameters())
        )
        optimizer = AdamW([
            {"params": pretrained_params, "lr": args.lr * 0.1, "initial_lr": args.lr * 0.1},
            {"params": new_params,        "lr": args.lr,       "initial_lr": args.lr},
        ], weight_decay=1e-4)
        print(
            f"  Optimizer: pretrained params @ lr={args.lr * 0.1:.2e}, "
            f"new params @ lr={args.lr:.2e}"
        )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(train_loader), 1), eta_min=1e-6
    )
    scaler = GradScaler(device.type if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

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
        if args.reset_lr:
            for g in optimizer.param_groups:
                g["lr"] = args.lr
                g["initial_lr"] = args.lr
            print(f"LR reset to {args.lr}")

    snr_range = (args.snr_min, args.snr_max)

    # ── loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # Optionally freeze pretrained components for early epochs
        if args.use_pretrained and args.freeze_pretrained_epochs > 0:
            frozen = epoch < args.freeze_pretrained_epochs
            for p in (*model.encoder.parameters(), *model.pitch_head.parameters()):
                p.requires_grad_(not frozen)
            if epoch == 0:
                print(f"  Freezing encoder+pitch_head for first {args.freeze_pretrained_epochs} epochs")
            elif epoch == args.freeze_pretrained_epochs:
                print(f"  Epoch {epoch:03d}: unfreezing encoder+pitch_head")

        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            cosine_scheduler, scaler, device, global_step, wandb_run,
            npz_loader=npz_loader, snr_range=snr_range,
        )

        val_metrics = evaluate_model(model, val_loader, device)
        val_pitch_loss = val_metrics.get("pitch_loss", float("nan"))

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_losses['total_loss']:.4f} | "
            f"pitch={val_metrics.get('pitch_rpa', 0.0) * 100:.1f}% "
            f"rhy={val_metrics.get('onset_f1', 0.0) * 100:.1f}% "
            f"bth={val_metrics.get('breath_acc', 0.0) * 100:.1f}% "
            f"vad={val_metrics.get('vad_acc', 0.0) * 100:.1f}% "
            f"overall={val_metrics.get('overall', 0.0) * 100:.1f}%"
        )
        if epoch % 5 == 0 or epoch == start_epoch:
            _print_eval_table(epoch, train_losses, val_metrics)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **{f"val/{k}": v for k, v in val_metrics.items()}})

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_payload = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_loss":             val_pitch_loss,
            "config":               vars(args),
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
