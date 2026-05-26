"""Coaching-quality evaluation script for VocalStars.

Runs the full ``analyse_recording()`` pipeline on a random sample of test-split
clips (using NPZ ground truth, no audio needed) and prints:

  - Aggregate metrics: pitch accuracy, drift, phrase length, onset clarity,
    technique accuracy
  - Sample coaching outputs for a handful of clips
  - A coaching score distribution (histogram)

This is what a backend call will eventually run — verifying that the model
produces coaching-quality output before wiring it in.

Usage::

    python ml_new/training/evaluate_coaching.py \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --checkpoint ml_new/checkpoints/unified/best.pt

    # With fine-tuned technique head:
    python ml_new/training/evaluate_coaching.py \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --checkpoint ml_new/checkpoints/unified_tech/best.pt

    # Limit to 50 clips for a quick check:
    python ml_new/training/evaluate_coaching.py \\
        --manifest ml_new/data/extracted_pyin/manifest.csv \\
        --checkpoint ml_new/checkpoints/unified/best.pt --n-clips 50
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.data.unified_dataset import _load_manifest, _filter_by_split
from ml_new.models.unified_model import (
    UnifiedVocalModel, TECHNIQUE_VOCAB, TECHNIQUE_TO_IDX,
)
from ml_new.inference.coach_inference import (
    _pitch_accuracy, _pitch_drift_cents, _phrase_lengths_s,
    _onset_clarity, _build_coaching_text, _build_result,
    VOICED_THRESH, BREATH_THRESH, ONSET_THRESH, HOP_S,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NPZ-based inference (no audio re-read — uses stored ground-truth arrays)
# ---------------------------------------------------------------------------

def _eval_clip_npz(
    row: dict,
    model: UnifiedVocalModel,
    device: torch.device,
) -> dict | None:
    """Run model on a single NPZ clip and return metric dict."""
    try:
        data = np.load(row["npz_path"], allow_pickle=False)
    except Exception as exc:
        log.debug("Skip %s: %s", row["npz_path"], exc)
        return None

    hcqt      = data["hcqt"]           # (6, 180, T)
    vad_feats = data["vad_features"]   # (3, T)
    f0_gt     = data["f0_hz"]          # ground-truth F0
    vad_gt    = data["vad"]

    T = min(hcqt.shape[2], vad_feats.shape[1], len(f0_gt), len(vad_gt))
    hcqt      = hcqt[:, :, :T]
    vad_feats = vad_feats[:, :T]
    f0_gt     = f0_gt[:T]
    vad_gt    = vad_gt[:T]

    hcqt_t   = torch.from_numpy(hcqt).unsqueeze(0).to(device)
    vad_t    = torch.from_numpy(vad_feats).unsqueeze(0).to(device)

    with torch.no_grad():
        pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, _ = \
            model(hcqt_t, vad_t)

    bin_hz_np = model.bin_hz.cpu().numpy()
    pitch_bins = pitch_logits[0].argmax(dim=-1).cpu().numpy()
    pitch_hz   = bin_hz_np[pitch_bins].astype(np.float32)
    voiced_np  = voiced_prob[0].cpu().numpy()
    breath_np  = breath_prob[0].cpu().numpy()
    onset_np   = onset_prob[0].cpu().numpy()

    voiced_bool = voiced_np >= VOICED_THRESH
    pitch_hz    = np.where(voiced_bool, pitch_hz, 0.0).astype(np.float32)
    breath_bool = breath_np >= BREATH_THRESH
    onset_bool  = onset_np  >= ONSET_THRESH

    # Technique
    tech_probs = torch.softmax(tech_logits[0], dim=-1).cpu().numpy()
    pred_idx   = int(np.argmax(tech_probs))
    gt_tech    = row.get("technique", "")
    gt_idx     = TECHNIQUE_TO_IDX.get(gt_tech, -1)

    # Pitch accuracy vs ground-truth F0 (not vs nearest semitone)
    voiced_gt  = (f0_gt > 0) & voiced_bool
    if voiced_gt.any():
        cents_err = 1200.0 * np.log2(
            (pitch_hz[voiced_gt] + 1e-8) / (f0_gt[voiced_gt] + 1e-8)
        )
        rpa_gt = float((np.abs(cents_err) < 50.0).mean())
    else:
        rpa_gt = float("nan")

    # Coaching metrics (self-consistent, not vs ground truth)
    pitch_acc   = _pitch_accuracy(pitch_hz, voiced_bool)
    drift_cents = _pitch_drift_cents(pitch_hz, voiced_bool)
    phrases     = _phrase_lengths_s(voiced_bool, HOP_S)
    onset_clar  = _onset_clarity(onset_np, onset_bool)
    onset_cnt   = int(np.diff(onset_bool.astype(np.int8), prepend=0).clip(min=0).sum())
    breath_cnt  = int(
        np.diff(breath_bool.astype(np.int8), prepend=0).clip(min=0).sum()
    )

    score, summary, issues, exercises = _build_coaching_text(
        pitch_acc, drift_cents, phrases, onset_clar, onset_cnt,
        TECHNIQUE_VOCAB[pred_idx], float(tech_probs[pred_idx]),
    )

    return {
        "npz_path":    row["npz_path"],
        "dataset":     row.get("dataset", ""),
        "singer_id":   row.get("singer_id", ""),
        "gt_technique": gt_tech,
        "pred_technique": TECHNIQUE_VOCAB[pred_idx],
        "tech_correct": (pred_idx == gt_idx) if gt_idx >= 0 else None,
        "rpa_gt":       rpa_gt,
        "pitch_acc":    pitch_acc,
        "drift_cents":  drift_cents,
        "mean_phrase_s": float(np.mean(phrases)) if phrases else 0.0,
        "n_phrases":    len(phrases),
        "breath_count": breath_cnt,
        "onset_count":  onset_cnt,
        "onset_clarity": onset_clar,
        "score":        score,
        "summary":      summary,
        "issues":       issues,
        "exercises":    exercises,
    }


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------

def _print_aggregate(results: list[dict]) -> None:
    def nanmean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    rpa_vals  = [r["rpa_gt"] for r in results]
    acc_vals  = [r["pitch_acc"] for r in results]
    drift_vals = [r["drift_cents"] for r in results]
    phrase_vals = [r["mean_phrase_s"] for r in results]
    clarity_vals = [r["onset_clarity"] for r in results]
    score_vals  = [r["score"] for r in results]

    tech_preds = [r for r in results if r["tech_correct"] is not None]
    tech_acc = float(np.mean([r["tech_correct"] for r in tech_preds])) if tech_preds else float("nan")

    print("\n" + "═" * 58)
    print("  Coaching Evaluation — Aggregate Metrics")
    print("═" * 58)
    print(f"  Clips evaluated        : {len(results)}")
    print(f"  Pitch RPA (vs GT F0)   : {nanmean(rpa_vals):.1%}   (target 90%)")
    print(f"  Pitch acc (vs semitone): {nanmean(acc_vals):.1%}")
    print(f"  Pitch drift            : {nanmean(drift_vals):+.1f} ¢"
          "  (0 = on-pitch, − = flat, + = sharp)")
    print(f"  Avg phrase length      : {nanmean(phrase_vals):.1f} s")
    print(f"  Onset clarity          : {nanmean(clarity_vals):.3f}")
    print(f"  Technique accuracy     : {tech_acc:.1%}   (needs fine-tune if < 65%)")
    print(f"  Avg coaching score     : {nanmean(score_vals):.0f}/100")

    # Score histogram (5 buckets)
    buckets = [0] * 5
    for s in score_vals:
        buckets[min(4, int(s) // 20)] += 1
    print("\n  Score distribution:")
    labels = ["0–19", "20–39", "40–59", "60–79", "80–100"]
    for lbl, cnt in zip(labels, buckets):
        bar = "█" * (cnt * 30 // max(1, len(results)))
        print(f"    {lbl}  {bar} {cnt}")


def _print_samples(results: list[dict], n: int = 5) -> None:
    sample = random.sample(results, min(n, len(results)))
    print("\n" + "─" * 58)
    print(f"  Sample Coaching Outputs ({len(sample)} clips)")
    print("─" * 58)
    for r in sample:
        clip_name = Path(r["npz_path"]).stem
        print(f"\n  Clip: {clip_name}  [{r['dataset']} / {r['singer_id']}]")
        print(f"  Technique: {r['pred_technique']} "
              f"(GT: {r['gt_technique']}, "
              f"{'✓' if r['tech_correct'] else '✗' if r['tech_correct'] is not None else '?'})")
        print(f"  Score: {r['score']}/100  |  {r['summary']}")
        for i, (issue, ex) in enumerate(zip(r["issues"], r["exercises"]), 1):
            print(f"    {i}. {issue}")
            print(f"       → {ex}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate coaching quality on the test split."
    )
    p.add_argument(
        "--manifest", type=Path,
        default=Path("ml_new/data/extracted_pyin/manifest.csv"),
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=Path("ml_new/checkpoints/unified/best.pt"),
    )
    p.add_argument("--n-clips",   type=int,   default=200,
                   help="Max clips to evaluate (0 = all in test split).")
    p.add_argument("--val-frac",  type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--n-samples", type=int,   default=5,
                   help="Sample coaching outputs to print.")
    p.add_argument("--device",    type=str,   default=None)
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
    if args.checkpoint.exists():
        ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        log.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        log.warning("Checkpoint not found — using random weights: %s", args.checkpoint)
    model.eval()

    # ── Test split ────────────────────────────────────────────────────────
    rows = _load_manifest(args.manifest)
    rows = _filter_by_split(rows, "test", args.val_frac, args.test_frac, args.seed)
    log.info("Test split: %d clips", len(rows))

    if args.n_clips > 0 and len(rows) > args.n_clips:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.n_clips)
        log.info("Subsampled to %d clips", len(rows))

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []
    for i, row in enumerate(rows):
        if (i + 1) % 50 == 0:
            log.info("  %d / %d", i + 1, len(rows))
        r = _eval_clip_npz(row, model, device)
        if r is not None:
            results.append(r)

    log.info("Evaluated %d / %d clips successfully", len(results), len(rows))

    _print_aggregate(results)
    _print_samples(results, n=args.n_samples)
    print()


if __name__ == "__main__":
    main()
