"""Train the unified multi-task vocal model.

Targets (90 %+ on all tasks):
  Pitch  : RPA   ≥ 90 %   (Raw Pitch Accuracy, within 50 ¢)
  Voiced : VAD F1 ≥ 90 %
  Breath : Recall ≥ 90 %  (@ threshold 0.35; catching breaths matters more)
  Onset  : Recall ≥ 90 %  (@ threshold 0.30; onset detection is noisy)
  Tech   : Accuracy ≥ 90 %

Loss
----
  pitch_loss   — Gaussian soft-label cross-entropy on voiced frames
  voiced_loss  — Focal BCE  (γ=2, pos_weight=2)
  breath_loss  — Focal BCE  (γ=3, pos_weight=15)  ← high weight for rare class
  onset_loss   — Focal BCE  (γ=3, pos_weight=12)  ← rare + noisy labels
  tech_loss    — Weighted cross-entropy (inverse-frequency class weights)

  total = 1.0·pitch + 0.6·voiced + 0.9·breath + 0.9·onset + 0.4·tech

Usage::

    python ml_new/training/train_unified.py \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --output-dir ml_new/checkpoints/unified \\
        --epochs 70 --batch-size 32 --lr 1e-3
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

from ml_new.data.unified_dataset import UnifiedDataset, technique_class_weights, TECHNIQUE_UNKNOWN
from ml_new.models.unified_model import UnifiedVocalModel, TECHNIQUE_VOCAB, N_TECHNIQUES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FMIN = UnifiedVocalModel.FMIN


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gaussian_pitch_loss(
    pitch_logits: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced_probs: torch.Tensor,
    sigma: float = 1.0,
    bins_per_octave: int = 36,
) -> torch.Tensor:
    """Cross-entropy with Gaussian soft labels on voiced frames."""
    voiced_mask = f0_hz > 0
    if not voiced_mask.any():
        return pitch_logits.sum() * 0.0

    B, T, n_bins = pitch_logits.shape
    device = pitch_logits.device

    f0_safe = f0_hz.clamp(min=1.0)
    bin_float = bins_per_octave * torch.log2(f0_safe / FMIN)
    bins = torch.arange(n_bins, dtype=torch.float32, device=device)
    diff = bin_float.unsqueeze(-1) - bins
    gauss = torch.exp(-0.5 * (diff / sigma) ** 2)
    gauss = gauss / gauss.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    gauss = gauss * voiced_mask.unsqueeze(-1)

    log_probs = F.log_softmax(pitch_logits, dim=-1)
    ce = -(gauss * log_probs).sum(dim=-1)
    return (ce * voiced_probs)[voiced_mask].mean()


def focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """Focal binary cross-entropy."""
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    bce_pos = -torch.log(pred)
    bce_neg = -torch.log(1 - pred)
    focal_pos = (1 - pred) ** gamma * bce_pos
    focal_neg = pred ** gamma * bce_neg
    loss = target * pos_weight * focal_pos + (1 - target) * focal_neg
    return loss.mean()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: UnifiedVocalModel,
    loader: DataLoader,
    device: torch.device,
    bins_per_octave: int = 36,
    breath_thresh: float = 0.35,
    onset_thresh: float = 0.30,
    voiced_thresh: float = 0.50,
) -> dict[str, float]:
    """Compute all five task metrics on a data split."""
    model.eval()

    # Pitch / voiced accumulators
    tp_v = fn_v = fp_v = tn_v = 0
    rpa_num = rca_num = voiced_denom = 0
    cents_errors: list[float] = []

    # Breath accumulators
    tp_b = fn_b = fp_b = tn_b = 0

    # Onset accumulators
    tp_o = fn_o = fp_o = tn_o = 0

    # Technique accumulators
    tech_correct = tech_total = 0

    # Loss tracking
    total_loss = n_batches = 0

    for batch in loader:
        hcqt       = batch["hcqt"].to(device)
        vad_feats  = batch["vad_features"].to(device)
        f0_hz      = batch["f0_hz"].to(device)
        vad_tgt    = batch["vad"].to(device)
        breath_tgt = batch["breath"].to(device)
        onset_tgt  = batch["onset"].to(device)
        tech_tgt   = batch["technique_idx"].to(device)
        vp         = batch["voiced_probs"].to(device)

        pl, vp_out, bp_out, op_out, tl_out, _ = model(hcqt, vad_feats)

        # ── voiced metrics ────────────────────────────────────────────────
        pred_v = (vp_out > voiced_thresh).float()
        gt_v   = (f0_hz > 0).float()
        tp_v += int(( pred_v *  gt_v).sum())
        fn_v += int(((1-pred_v) *  gt_v).sum())
        fp_v += int(( pred_v * (1-gt_v)).sum())
        tn_v += int(((1-pred_v) * (1-gt_v)).sum())

        # ── pitch metrics ─────────────────────────────────────────────────
        both_voiced = (gt_v.bool()) & (pred_v.bool())
        if both_voiced.any():
            pred_bins = pl.argmax(dim=-1)                     # (B, T)
            pred_hz = FMIN * (2.0 ** (pred_bins.float() / bins_per_octave))
            gt_hz = f0_hz.clamp(min=1e-3)

            pred_cents = 1200.0 * torch.log2(pred_hz[both_voiced] / gt_hz[both_voiced])
            abs_cents  = pred_cents.abs()
            abs_chroma = torch.min(abs_cents % 1200.0, 1200.0 - abs_cents % 1200.0)

            n_voiced = int(gt_v.sum())
            rpa_num  += int((abs_cents  < 50).sum())
            rca_num  += int((abs_chroma < 50).sum())
            voiced_denom += n_voiced
            cents_errors.extend(abs_cents.cpu().tolist())

        # ── breath metrics ────────────────────────────────────────────────
        pred_b = (bp_out > breath_thresh).float()
        gt_b   = breath_tgt
        tp_b += int(( pred_b *  gt_b).sum())
        fn_b += int(((1-pred_b) *  gt_b).sum())
        fp_b += int(( pred_b * (1-gt_b)).sum())
        tn_b += int(((1-pred_b) * (1-gt_b)).sum())

        # ── onset metrics ─────────────────────────────────────────────────
        pred_o = (op_out > onset_thresh).float()
        gt_o   = onset_tgt
        tp_o += int(( pred_o *  gt_o).sum())
        fn_o += int(((1-pred_o) *  gt_o).sum())
        fp_o += int(( pred_o * (1-gt_o)).sum())
        tn_o += int(((1-pred_o) * (1-gt_o)).sum())

        # ── technique metrics ─────────────────────────────────────────────
        known = tech_tgt >= 0
        if known.any():
            pred_tech = tl_out[known].argmax(dim=-1)
            tech_correct += int((pred_tech == tech_tgt[known]).sum())
            tech_total   += int(known.sum())

        n_batches += 1

    def safe_div(a: int, b: int) -> float:
        return a / b if b > 0 else 0.0

    vdr  = safe_div(tp_v, tp_v + fn_v)
    vfa  = safe_div(fp_v, fp_v + tn_v)
    rpa  = safe_div(rpa_num, voiced_denom)
    rca  = safe_div(rca_num, voiced_denom)
    mc   = float(torch.tensor(cents_errors).median()) if cents_errors else 0.0

    # Voiced F1
    prec_v = safe_div(tp_v, tp_v + fp_v)
    rec_v  = safe_div(tp_v, tp_v + fn_v)
    f1_v   = safe_div(2 * prec_v * rec_v, prec_v + rec_v)

    # Breath F1 / recall
    prec_b = safe_div(tp_b, tp_b + fp_b)
    rec_b  = safe_div(tp_b, tp_b + fn_b)
    f1_b   = safe_div(2 * prec_b * rec_b, prec_b + rec_b)

    # Onset F1 / recall
    prec_o = safe_div(tp_o, tp_o + fp_o)
    rec_o  = safe_div(tp_o, tp_o + fn_o)
    f1_o   = safe_div(2 * prec_o * rec_o, prec_o + rec_o)

    tech_acc = safe_div(tech_correct, tech_total)

    return {
        "rpa": rpa, "rca": rca,
        "vdr": vdr, "vfa": vfa, "vad_f1": f1_v,
        "breath_recall": rec_b, "breath_prec": prec_b, "breath_f1": f1_b,
        "onset_recall":  rec_o, "onset_prec":  prec_o, "onset_f1":  f1_o,
        "tech_acc": tech_acc,
        "median_cents": mc,
    }


def composite_score(m: dict[str, float]) -> float:
    """Single number for checkpoint selection (higher = better).

    Normalises each metric relative to the 90 % target so the score reaches
    1.0 when all targets are hit.  Pitch gets double weight.
    """
    return (
        2.0 * m["rpa"]         / 0.90 +
        1.0 * m["vad_f1"]      / 0.90 +
        1.0 * m["breath_recall"] / 0.90 +
        1.0 * m["onset_recall"]  / 0.90 +
        1.0 * m["tech_acc"]    / 0.90
    ) / 6.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log.info("device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = UnifiedDataset(
        args.manifest, seq_len=args.seq_len, split="train",
        augment=True, smooth_f0=True,
    )
    val_ds = UnifiedDataset(
        args.manifest, seq_len=args.seq_len, split="val",
    )
    log.info("train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = UnifiedVocalModel(gru_hidden=128, num_gru_layers=2).to(device)
    log.info("params: %s", f"{model.param_count():,}")

    # ── Class weights for technique head ──────────────────────────────────
    tech_weights = technique_class_weights(args.manifest).to(device)
    log.info("technique weights: min=%.2f  max=%.2f", tech_weights.min(), tech_weights.max())

    # ── Optimiser + schedule ──────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── CSV logger ────────────────────────────────────────────────────────
    log_path = output_dir / "train_log.csv"
    log_fields = [
        "epoch", "train_loss", "val_loss",
        "rpa", "rca", "vdr", "vfa", "vad_f1",
        "breath_recall", "breath_f1",
        "onset_recall", "onset_f1",
        "tech_acc", "median_cents", "composite", "lr",
    ]
    log_fh = open(log_path, "w", newline="")
    log_csv = csv.DictWriter(log_fh, fieldnames=log_fields)
    log_csv.writeheader()

    best_score = -1.0
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            hcqt       = batch["hcqt"].to(device)
            vad_feats  = batch["vad_features"].to(device)
            f0_hz      = batch["f0_hz"].to(device)
            vad_tgt    = batch["vad"].to(device)
            breath_tgt = batch["breath"].to(device)
            onset_tgt  = batch["onset"].to(device)
            tech_tgt   = batch["technique_idx"].to(device)
            vp         = batch["voiced_probs"].to(device)

            pl, vp_out, bp_out, op_out, tl_out, _ = model(hcqt, vad_feats)

            # ── Individual task losses ─────────────────────────────────────
            l_pitch = gaussian_pitch_loss(
                pl, f0_hz, vp, sigma=1.0, bins_per_octave=36
            )
            l_voiced = focal_bce(vp_out, (f0_hz > 0).float(), gamma=2.0, pos_weight=2.0)
            l_breath = focal_bce(bp_out, breath_tgt, gamma=3.0, pos_weight=15.0)
            l_onset  = focal_bce(op_out, onset_tgt,  gamma=3.0, pos_weight=12.0)

            # Technique: only compute on clips with a known technique label
            known = tech_tgt >= 0
            if known.any():
                l_tech = F.cross_entropy(
                    tl_out[known], tech_tgt[known], weight=tech_weights
                )
            else:
                l_tech = l_pitch * 0.0

            loss = (
                1.0 * l_pitch
                + 0.6 * l_voiced
                + 0.9 * l_breath
                + 0.9 * l_onset
                + 0.4 * l_tech
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # Warmup then cosine
            if step < warmup_steps:
                warmup_sched.step()
            else:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            step       += 1

        avg_train_loss = epoch_loss / max(1, n_batches)

        # ── Validation ────────────────────────────────────────────────────
        m = evaluate(model, val_loader, device)
        score = composite_score(m)
        lr_now = opt.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────
        log.info(
            "ep %02d | loss=%.4f | "
            "RPA=%.1f%% VAD_F1=%.1f%% "
            "B_rec=%.1f%% O_rec=%.1f%% Tech=%.1f%% | "
            "composite=%.3f",
            epoch, avg_train_loss,
            m["rpa"]*100, m["vad_f1"]*100,
            m["breath_recall"]*100, m["onset_recall"]*100,
            m["tech_acc"]*100, score,
        )

        log_csv.writerow({
            "epoch": epoch,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": "",
            "rpa":    f"{m['rpa']:.6f}",
            "rca":    f"{m['rca']:.6f}",
            "vdr":    f"{m['vdr']:.6f}",
            "vfa":    f"{m['vfa']:.6f}",
            "vad_f1": f"{m['vad_f1']:.6f}",
            "breath_recall": f"{m['breath_recall']:.6f}",
            "breath_f1":     f"{m['breath_f1']:.6f}",
            "onset_recall":  f"{m['onset_recall']:.6f}",
            "onset_f1":      f"{m['onset_f1']:.6f}",
            "tech_acc":      f"{m['tech_acc']:.6f}",
            "median_cents":  f"{m['median_cents']:.3f}",
            "composite":     f"{score:.6f}",
            "lr": f"{lr_now:.3e}",
        })
        log_fh.flush()

        # ── Checkpoint ────────────────────────────────────────────────────
        torch.save(model.state_dict(), output_dir / "latest.pt")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), output_dir / "best.pt")
            log.info("  ✓ new best (score=%.4f)", score)

        # ── Print per-task status vs targets ──────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs:
            _print_targets(m)

    log_fh.close()
    log.info("Done. Best composite score: %.4f", best_score)
    log.info("Checkpoint: %s", output_dir / "best.pt")


def _print_targets(m: dict[str, float]) -> None:
    rows = [
        ("Pitch  RPA",     m["rpa"],           0.90),
        ("VAD    F1",      m["vad_f1"],         0.90),
        ("Breath Recall",  m["breath_recall"],  0.90),
        ("Onset  Recall",  m["onset_recall"],   0.90),
        ("Tech   Acc",     m["tech_acc"],       0.90),
    ]
    log.info("  %-18s  %-8s  %-8s  %s", "Task", "Current", "Target", "")
    for name, val, tgt in rows:
        marker = "✓" if val >= tgt else "…"
        log.info("  %-18s  %-8s  %-8s  %s", name, f"{val*100:.1f}%", f"{tgt*100:.0f}%", marker)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified multi-task vocal model")
    parser.add_argument(
        "--manifest",
        default="ml_new/data/extracted_pyin/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_new/checkpoints/unified",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--epochs",      type=int,   default=70)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seq-len",     type=int,   default=200)
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
