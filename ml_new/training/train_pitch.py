"""Training script for the pitch detection model.

Trains a PitchModel on pre-extracted NPZ features.

Loss
----
- **Pitch loss** (voiced frames only): cross-entropy with Gaussian soft labels.
  A Gaussian centred on the true pitch bin (sigma=1.0 bin = 100 cents) is used
  as the target distribution, allowing the model to predict adjacent bins and
  still receive partial credit.
- **Voiced loss**: focal BCE (same as VAD model) on all frames.
- **Total**: pitch_weight * pitch_loss + voiced_weight * voiced_loss

Metrics
-------
- RPA  — Raw Pitch Accuracy: % voiced frames with |pred − true| < 50 cents
- RCA  — Raw Chroma Accuracy: octave-invariant RPA (mod 1200 cents)
- VDR  — Voicing Detection Rate (recall of voiced frames)
- VFA  — Voicing False Alarm Rate (false positives on unvoiced frames)
- Median cents error on voiced frames

Usage::

    python ml_new/training/train_pitch.py \\
        --manifest ml_new/data/extracted/manifest.csv \\
        --output-dir ml_new/checkpoints/pitch \\
        --epochs 40 --batch-size 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.data.feature_dataset import FeatureDataset
from ml_new.models.pitch_model import PitchModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FMIN = PitchModel.FMIN


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def gaussian_pitch_loss(
    pitch_logits: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced_probs: torch.Tensor | None = None,
    sigma: float = 1.0,
    bins_per_octave: int = 36,
) -> torch.Tensor:
    """Cross-entropy loss with Gaussian soft labels on voiced frames.

    Args:
        pitch_logits: ``(B, T, n_bins)`` unnormalized logits.
        f0_hz: ``(B, T)`` ground-truth F0 in Hz; 0.0 = unvoiced.
        voiced_probs: ``(B, T)`` per-frame confidence weights in [0, 1].
            From pyin's voiced probability; 1.0 for yin-extracted frames.
            Low-confidence frames (vibrato peaks, falsetto onset) contribute
            less to the loss, reducing the impact of noisy labels.
        sigma: Label smoothing width in bins.
        bins_per_octave: CQT resolution, used to convert Hz → bin index.

    Returns:
        Scalar mean loss over voiced frames, or 0 if no voiced frames.
    """
    voiced_mask = f0_hz > 0
    if not voiced_mask.any():
        return torch.tensor(0.0, device=pitch_logits.device, requires_grad=True)

    device = pitch_logits.device
    B, T, n_bins = pitch_logits.shape

    # Continuous bin index for each voiced frame
    f0_safe = f0_hz.clamp(min=1.0)
    bin_float = bins_per_octave * torch.log2(f0_safe / FMIN)   # (B, T)

    # Gaussian soft targets: (B, T, n_bins)
    bins = torch.arange(n_bins, dtype=torch.float32, device=device)
    diff = bin_float.unsqueeze(-1) - bins                       # (B, T, n_bins)
    gauss = torch.exp(-0.5 * (diff / sigma) ** 2)
    gauss = gauss / gauss.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Mask out-of-range unvoiced targets (shouldn't contribute to loss)
    gauss = gauss * voiced_mask.unsqueeze(-1)

    # Cross-entropy: -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(pitch_logits, dim=-1)             # (B, T, n_bins)
    ce = -(gauss * log_probs).sum(dim=-1)                       # (B, T)

    if voiced_probs is not None:
        return (ce * voiced_probs)[voiced_mask].mean()
    return ce[voiced_mask].mean()


def focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """Focal binary cross-entropy (same formulation as in train_vad.py)."""
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    bce = -(pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    pt = torch.where(target == 1, pred, 1 - pred)
    return ((1 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pitch_metrics(
    pitch_logits: torch.Tensor,
    voiced_prob: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced_threshold: float = 0.5,
    bins_per_octave: int = 36,
) -> dict[str, float]:
    """Compute pitch and voicing accuracy metrics.

    Args:
        pitch_logits: ``(B, T, 60)`` logits.
        voiced_prob: ``(B, T)`` predicted voicing probability.
        f0_hz: ``(B, T)`` ground-truth F0; 0.0 = unvoiced.
        voiced_threshold: Decision boundary for voiced/unvoiced.

    Returns:
        Dict with keys: rpa, rca, vdr, vfa, median_cents.
    """
    true_voiced = f0_hz > 0
    pred_voiced = voiced_prob >= voiced_threshold

    # Voicing metrics
    tp_v = (pred_voiced & true_voiced).sum().float()
    fn_v = (~pred_voiced & true_voiced).sum().float()
    fp_v = (pred_voiced & ~true_voiced).sum().float()
    tn_v = (~pred_voiced & ~true_voiced).sum().float()
    vdr = (tp_v / (tp_v + fn_v + 1e-9)).item()
    vfa = (fp_v / (fp_v + tn_v + 1e-9)).item()

    # Pitch accuracy on truly voiced frames
    if not true_voiced.any():
        return {"rpa": 0.0, "rca": 0.0, "vdr": vdr, "vfa": vfa, "median_cents": 0.0}

    pred_bins = pitch_logits.argmax(dim=-1)  # (B, T)
    pred_hz = FMIN * (2.0 ** (pred_bins.float() / bins_per_octave))

    true_hz_voiced = f0_hz[true_voiced].clamp(min=1.0)
    pred_hz_voiced = pred_hz[true_voiced].clamp(min=1.0)

    cents_err = (1200.0 * torch.log2(pred_hz_voiced / true_hz_voiced)).abs()

    rpa = (cents_err < 50.0).float().mean().item()
    cents_chroma = cents_err % 1200.0
    cents_chroma = torch.minimum(cents_chroma, 1200.0 - cents_chroma)
    rca = (cents_chroma < 50.0).float().mean().item()
    median_cents = cents_err.median().item()

    return {
        "rpa": rpa,
        "rca": rca,
        "vdr": vdr,
        "vfa": vfa,
        "median_cents": median_cents,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    manifest: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seq_len: int = 200,
    pitch_weight: float = 1.0,
    voiced_weight: float = 0.5,
    focal_gamma: float = 2.0,
    voiced_pos_weight: float = 2.0,
    pitch_sigma: float = 1.0,
    gru_hidden: int = 96,
    num_layers: int = 1,
    dropout: float = 0.1,
    n_bins: int = 180,
    bins_per_octave: int = 36,
    augment: bool = False,
    shift_semitones: int = 3,
    smooth_f0: bool = False,
    use_voiced_probs: bool = False,
    init_checkpoint: Path | None = None,
    device_str: str = "auto",
) -> None:
    """Run the full pitch model training loop.

    Args:
        manifest: Path to manifest.csv.
        output_dir: Directory for checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        lr: Initial AdamW learning rate.
        seq_len: Frames per training window (200 ≈ 2 s).
        pitch_weight: Weight of pitch CE loss in total loss.
        voiced_weight: Weight of voiced focal BCE loss.
        focal_gamma: Focal loss exponent for voicing.
        voiced_pos_weight: Positive-class weight for voicing loss.
        pitch_sigma: Gaussian soft-label width in bins.
        gru_hidden: GRU hidden dimension.
        num_layers: GRU layer count.
        dropout: Dropout between GRU layers.
        n_bins: CQT bins per harmonic layer.
        bins_per_octave: CQT frequency resolution.
        augment: Enable pitch-shift augmentation for training split.
        shift_semitones: Max semitone shift for augmentation.
        smooth_f0: Apply 5-frame median filter to f0 labels (removes YIN spikes).
        use_voiced_probs: Weight pitch loss by pyin voiced_probs. Off by default;
            only meaningful when training on pyin-extracted data.
        init_checkpoint: If given, initialise model weights from this checkpoint
            before training (optimizer starts fresh — useful for fine-tuning).
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

    train_ds = FeatureDataset(
        manifest, seq_len=seq_len, split="train", task="both",
        augment=augment, shift_semitones=shift_semitones,
        bins_per_octave=bins_per_octave, smooth_f0=smooth_f0,
    )
    val_ds = FeatureDataset(manifest, seq_len=seq_len, split="val", task="both")
    log.info("Train: %d clips | Val: %d clips", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        persistent_workers=True,
    )

    model = PitchModel(
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        gru_hidden=gru_hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    log.info("PitchModel parameters: %d", model.param_count())

    if init_checkpoint is not None:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        log.info("Initialized weights from %s (epoch %d, val_rpa=%.4f)",
                 init_checkpoint, ckpt["epoch"], ckpt.get("val_rpa", 0.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    best_rpa = 0.0

    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "rpa", "rca", "vdr", "vfa", "median_cents", "lr",
    ]
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                hcqt = batch["hcqt"].to(device)
                vad_feats = batch["vad_features"].to(device)
                f0_hz = batch["f0_hz"].to(device)
                vad_target = batch["vad"].to(device)

                pitch_logits, voiced_prob, _ = model(hcqt, vad_feats)

                vp = batch.get("voiced_probs") if use_voiced_probs else None
                if vp is not None:
                    vp = vp.to(device)
                p_loss = gaussian_pitch_loss(pitch_logits, f0_hz, voiced_probs=vp,
                                             sigma=pitch_sigma, bins_per_octave=bins_per_octave)
                v_loss = focal_bce(
                    voiced_prob, vad_target,
                    gamma=focal_gamma, pos_weight=voiced_pos_weight,
                )
                loss = pitch_weight * p_loss + voiced_weight * v_loss

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
            all_logits, all_voiced_prob, all_f0, all_vad = [], [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    hcqt = batch["hcqt"].to(device)
                    vad_feats = batch["vad_features"].to(device)
                    f0_hz = batch["f0_hz"].to(device)
                    vad_target = batch["vad"].to(device)

                    pitch_logits, voiced_prob, _ = model(hcqt, vad_feats)

                    p_loss = gaussian_pitch_loss(pitch_logits, f0_hz, sigma=pitch_sigma,
                                                 bins_per_octave=bins_per_octave)
                    v_loss = focal_bce(
                        voiced_prob, vad_target,
                        gamma=focal_gamma, pos_weight=voiced_pos_weight,
                    )
                    val_loss += (pitch_weight * p_loss + voiced_weight * v_loss).item()

                    all_logits.append(pitch_logits.cpu())
                    all_voiced_prob.append(voiced_prob.cpu())
                    all_f0.append(f0_hz.cpu())
                    all_vad.append(vad_target.cpu())

            val_loss /= max(len(val_loader), 1)

            metrics = pitch_metrics(
                torch.cat(all_logits),
                torch.cat(all_voiced_prob),
                torch.cat(all_f0),
                bins_per_octave=bins_per_octave,
            )

            current_lr = scheduler.get_last_lr()[0]
            log.info(
                "Epoch %d/%d | loss=%.4f val=%.4f | "
                "RPA=%.3f RCA=%.3f VDR=%.3f VFA=%.3f med=%.0f¢ | lr=%.2e",
                epoch, epochs, train_loss, val_loss,
                metrics["rpa"], metrics["rca"],
                metrics["vdr"], metrics["vfa"],
                metrics["median_cents"], current_lr,
            )

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "rpa": f"{metrics['rpa']:.6f}",
                "rca": f"{metrics['rca']:.6f}",
                "vdr": f"{metrics['vdr']:.6f}",
                "vfa": f"{metrics['vfa']:.6f}",
                "median_cents": f"{metrics['median_cents']:.1f}",
                "lr": f"{current_lr:.2e}",
            }
            writer.writerow(row)
            fh.flush()

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_rpa": metrics["rpa"],
                "config": {
                    "gru_hidden": gru_hidden,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "pitch_sigma": pitch_sigma,
                    "n_bins": n_bins,
                    "bins_per_octave": bins_per_octave,
                },
            }
            torch.save(ckpt, output_dir / "latest.pt")

            if metrics["rpa"] > best_rpa:
                best_rpa = metrics["rpa"]
                torch.save(ckpt, output_dir / "best.pt")
                log.info("  → New best RPA=%.4f saved.", best_rpa)

    log.info("Training complete. Best val RPA=%.4f", best_rpa)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pitch detection model.")
    p.add_argument("--manifest", type=Path, default=Path("ml_new/data/extracted/manifest.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("ml_new/checkpoints/pitch"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--gru-hidden", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pitch-weight", type=float, default=1.0)
    p.add_argument("--voiced-weight", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--voiced-pos-weight", type=float, default=2.0)
    p.add_argument("--pitch-sigma", type=float, default=1.0)
    p.add_argument("--n-bins", type=int, default=180)
    p.add_argument("--bins-per-octave", type=int, default=36)
    p.add_argument("--augment", action="store_true",
                   help="Enable pitch-shift augmentation on training split")
    p.add_argument("--shift-semitones", type=int, default=3,
                   help="Max semitone shift for pitch augmentation")
    p.add_argument("--smooth-f0", action="store_true",
                   help="Apply 5-frame median filter to f0 labels to remove YIN spikes")
    p.add_argument("--use-voiced-probs", action="store_true",
                   help="Weight pitch loss by pyin voiced_probs (only useful with pyin-extracted data)")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Initialise model weights from this checkpoint before training (optimizer resets)")
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
        pitch_weight=args.pitch_weight,
        voiced_weight=args.voiced_weight,
        focal_gamma=args.focal_gamma,
        voiced_pos_weight=args.voiced_pos_weight,
        pitch_sigma=args.pitch_sigma,
        gru_hidden=args.gru_hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        augment=args.augment,
        shift_semitones=args.shift_semitones,
        smooth_f0=args.smooth_f0,
        use_voiced_probs=args.use_voiced_probs,
        init_checkpoint=args.init_checkpoint,
        device_str=args.device,
    )
