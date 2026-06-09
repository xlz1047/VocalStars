"""Evaluate a pitch checkpoint with argmax and Viterbi decoding side-by-side.

Usage::

    python ml_new/training/evaluate_pitch.py \\
        --checkpoint ml_new/checkpoints/pitch/best.pt \\
        --manifest ml_new/data/extracted/manifest.csv \\
        --exp-name exp1_viterbi

    # For high-res checkpoint:
    python ml_new/training/evaluate_pitch.py \\
        --checkpoint ml_new/checkpoints/pitch_hires_aug/best.pt \\
        --manifest ml_new/data/extracted_hires/manifest.csv \\
        --n-bins 180 --bins-per-octave 36 \\
        --exp-name exp4_viterbi_hires
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.data.feature_dataset import FeatureDataset
from ml_new.models.pitch_model import PitchModel
from ml_new.models.viterbi import pitch_viterbi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _pitch_metrics_from_arrays(
    pred_hz: torch.Tensor,
    true_f0: torch.Tensor,
    pred_voiced: torch.Tensor,
    true_voiced: torch.Tensor,
) -> dict[str, float]:
    tp_v = (pred_voiced & true_voiced).sum().float()
    fn_v = (~pred_voiced & true_voiced).sum().float()
    fp_v = (pred_voiced & ~true_voiced).sum().float()
    tn_v = (~pred_voiced & ~true_voiced).sum().float()
    vdr = (tp_v / (tp_v + fn_v + 1e-9)).item()
    vfa = (fp_v / (fp_v + tn_v + 1e-9)).item()

    if not true_voiced.any():
        return {"rpa": 0.0, "rca": 0.0, "vdr": vdr, "vfa": vfa, "median_cents": 0.0}

    th = true_f0[true_voiced].clamp(min=1.0)
    ph = pred_hz[true_voiced].clamp(min=1.0)
    cents_err = (1200.0 * torch.log2(ph / th)).abs()

    rpa = (cents_err < 50.0).float().mean().item()
    chroma = cents_err % 1200.0
    chroma = torch.minimum(chroma, 1200.0 - chroma)
    rca = (chroma < 50.0).float().mean().item()
    return {
        "rpa": rpa,
        "rca": rca,
        "vdr": vdr,
        "vfa": vfa,
        "median_cents": cents_err.median().item(),
    }


def evaluate(
    checkpoint: Path,
    manifest: Path,
    exp_name: str,
    n_bins: int = 60,
    bins_per_octave: int = 12,
    batch_size: int = 32,
    seq_len: int = 200,
    sigma_bins: float = 2.0,
    voice_change_penalty: float = 5.0,
    voiced_threshold: float = 0.5,
    device_str: str = "auto",
) -> None:
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

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    gru_hidden = cfg.get("gru_hidden", 128)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.2)
    # Override n_bins/bins_per_octave from checkpoint if not explicitly set on CLI
    n_bins = cfg.get("n_bins", n_bins)
    bins_per_octave = cfg.get("bins_per_octave", bins_per_octave)

    model = PitchModel(
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        gru_hidden=gru_hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint: epoch %d  (val_rpa=%.4f)", ckpt["epoch"], ckpt.get("val_rpa", 0.0))

    fmin = PitchModel.FMIN
    bin_hz_np = fmin * (2.0 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))
    log.info("Resolution: %d bins/oct = %.1f¢/bin  |  %d total bins", bins_per_octave,
             1200.0 / bins_per_octave, n_bins)

    val_ds = FeatureDataset(manifest, seq_len=seq_len, split="val", task="both")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    log.info("Val clips: %d", len(val_ds))

    # Collect argmax, weighted-mean, and Viterbi predictions
    argmax_preds, wmean_preds, viterbi_preds = [], [], []
    true_voiced_all, true_f0_all = [], []

    bins_float = torch.arange(n_bins, dtype=torch.float32)  # (n_bins,)

    with torch.no_grad():
        for batch in val_loader:
            hcqt = batch["hcqt"].to(device)
            vad_feats = batch["vad_features"].to(device)
            f0_hz = batch["f0_hz"]       # (B, T) on CPU
            vad_target = batch["vad"]    # (B, T) uint8 on CPU

            pitch_logits, voiced_prob, _ = model(hcqt, vad_feats)
            pitch_logits_cpu = pitch_logits.cpu()
            voiced_prob_cpu = voiced_prob.cpu()

            pred_voiced_am = voiced_prob_cpu >= voiced_threshold

            # Argmax decoding
            pred_bins_am = pitch_logits_cpu.argmax(dim=-1)  # (B, T)
            pred_hz_am = bin_hz_np[pred_bins_am]            # (B, T)

            # Weighted-mean (center-of-mass) decoding — sub-bin precision
            probs = torch.softmax(pitch_logits_cpu, dim=-1)               # (B, T, n_bins)
            bin_mean = (probs * bins_float).sum(dim=-1)                   # (B, T) fractional bin
            pred_hz_wm = fmin * (2.0 ** (bin_mean / bins_per_octave))    # (B, T) Hz

            argmax_preds.append((pred_hz_am, pred_voiced_am))
            wmean_preds.append((pred_hz_wm, pred_voiced_am))
            true_voiced_all.append(f0_hz > 0)
            true_f0_all.append(f0_hz)

            # Viterbi decoding (per sequence in batch)
            batch_ph_vit = []
            batch_pv_vit = []
            for b in range(pitch_logits_cpu.shape[0]):
                pb, pv = pitch_viterbi(
                    pitch_logits_cpu[b],
                    voiced_prob_cpu[b],
                    sigma_bins=sigma_bins,
                    voice_change_penalty=voice_change_penalty,
                    voiced_threshold=voiced_threshold,
                    pitch_only=True,
                )
                ph_vit = torch.from_numpy(bin_hz_np.numpy()[pb.clip(0)])
                ph_vit[~torch.from_numpy(pv)] = 0.0
                batch_ph_vit.append(ph_vit)
                batch_pv_vit.append(torch.from_numpy(pv))
            viterbi_preds.append((
                torch.stack(batch_ph_vit),
                torch.stack(batch_pv_vit),
            ))

    # Concatenate all
    all_true_f0 = torch.cat(true_f0_all)
    all_true_voiced = torch.cat(true_voiced_all)

    am_hz = torch.cat([x[0] for x in argmax_preds])
    am_voiced = torch.cat([x[1] for x in argmax_preds])

    wm_hz = torch.cat([x[0] for x in wmean_preds])

    vit_hz = torch.cat([x[0] for x in viterbi_preds])
    vit_voiced = torch.cat([x[1] for x in viterbi_preds])

    argmax_metrics = _pitch_metrics_from_arrays(am_hz, all_true_f0, am_voiced, all_true_voiced)
    wmean_metrics = _pitch_metrics_from_arrays(wm_hz, all_true_f0, am_voiced, all_true_voiced)
    viterbi_metrics = _pitch_metrics_from_arrays(vit_hz, all_true_f0, vit_voiced, all_true_voiced)

    log.info("=" * 70)
    log.info("Experiment: %s", exp_name)
    log.info("%-20s  %-9s  %-9s  %-9s", "Metric", "Argmax", "WgtMean", "Viterbi")
    log.info("-" * 70)
    for key in ("rpa", "rca", "vdr", "vfa", "median_cents"):
        log.info("%-20s  %-9.4f  %-9.4f  %-9.4f",
                 key, argmax_metrics[key], wmean_metrics[key], viterbi_metrics[key])
    log.info("=" * 70)

    results = {
        "exp_name": exp_name,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "config": {
            "n_bins": n_bins,
            "bins_per_octave": bins_per_octave,
            "sigma_bins": sigma_bins,
            "voice_change_penalty": voice_change_penalty,
            "voiced_threshold": voiced_threshold,
        },
        "argmax": argmax_metrics,
        "weighted_mean": wmean_metrics,
        "viterbi": viterbi_metrics,
    }

    out_path = checkpoint.parent / f"results_{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pitch checkpoint: argmax vs Viterbi.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("ml_new/data/extracted/manifest.csv"))
    p.add_argument("--exp-name", type=str, default="exp1_viterbi")
    p.add_argument("--n-bins", type=int, default=60)
    p.add_argument("--bins-per-octave", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--sigma-bins", type=float, default=2.0)
    p.add_argument("--voice-change-penalty", type=float, default=5.0)
    p.add_argument("--voiced-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        exp_name=args.exp_name,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        sigma_bins=args.sigma_bins,
        voice_change_penalty=args.voice_change_penalty,
        voiced_threshold=args.voiced_threshold,
        device_str=args.device,
    )
