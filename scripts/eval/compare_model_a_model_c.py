#!/usr/bin/env python3
"""Compare Model A and Model C on shared samples and available labels.

This is report-only. Model A and Model C do not expose the same heads:

- Model A: pitch/VAD/breath/onset plus a 20-class clip-level technique softmax.
- Model C: frame-level six-label technique sigmoid heads, with `mix` hidden.

The comparison therefore reports:

- self-recorded WAV predictions with no technique accuracy claim.
- bounded labelled manifest clip checks where source labels overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.model_c.evaluate import load_model as load_model_c  # noqa: E402
from ml_new.model_c.labels import HIDDEN_CONTEXT_TECHNIQUES, TECHNIQUES, TECHNIQUE_TO_IDX, USER_FACING_TECHNIQUES  # noqa: E402
from ml_new.model_c.predict_manifest import predict_clip as predict_model_c_clip  # noqa: E402
from ml_new.model_c.predict_manifest import postprocess_technique_prediction  # noqa: E402
from ml_new.model_c.train import device_for_arg  # noqa: E402
from ml_new.feature_extraction.hcqt import HCQTExtractor  # noqa: E402
from ml_new.feature_extraction.vad_features import VADFeatureExtractor  # noqa: E402
from ml_new.inference.coach_inference import BINS_PER_OCTAVE, HOP_LENGTH, N_BINS, SR  # noqa: E402
from ml_new.models.unified_model import TECHNIQUE_VOCAB  # noqa: E402
from scripts.eval.evaluate_nanopitch_wav import DEFAULT_CHECKPOINT as NANOPITCH_CHECKPOINT, run_nanopitch  # noqa: E402
from scripts.eval.audit_model_outputs import load_audio, run_raw_checkpoint, summarize_sample  # noqa: E402


SELF_WAVS = [
    Path("samples/00_silence.wav"),
    Path("samples/01_speaking_voice.wav"),
    Path("samples/03_sustained_aaa.wav"),
    Path("samples/04_pitch_slide.wav"),
    Path("samples/05_twinkle_twinkle.wav"),
]

MODEL_A_TO_MODEL_C_SHARED = {
    "breathy": "breathy",
    "pharyngeal": "pharyngeal",
    "glissando": "glissando",
    "vibrato": "vibrato",
}
HIDDEN_CONTEXT_LABELS = {"mixed_voice_and_falsetto"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stratified_rows(rows: list[dict[str, str]], split: str, clips_per_label: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != split:
            continue
        if not Path(row.get("audio_path", "")).exists():
            continue
        if not Path(row.get("npz_path", "")).exists() or not Path(row.get("nanopitch_npz", "")).exists():
            continue
        grouped[row.get("source_technique", "")].append(row)
    selected: list[dict[str, str]] = []
    for label in sorted(grouped):
        selected.extend(grouped[label][:clips_per_label])
    return selected


def top_model_a_technique(audio_path: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    audio = load_audio(audio_path)
    raw = run_raw_checkpoint(audio, checkpoint, device)
    summary = summarize_sample(audio_path.stem, audio, raw)
    tech = summary["technique"]
    return {
        "top_label": tech["top_label"],
        "top_probability": tech["top_probability"],
        "probabilities": tech["probabilities"],
        "voiced_fraction": summary["voiced_probability"]["fraction_above_0_5"],
        "breath_probability_mean": summary["breath_probability"]["mean"],
        "onset_probability_mean": summary["onset_probability"]["mean"],
        "raw_f0_median_hz": summary["raw_f0_before_smoothing"]["median_hz"],
        "note": "Model A technique head is clip-level and marked unreliable in prior audits.",
    }


def summarize_model_c_prediction(record: dict[str, Any]) -> dict[str, Any]:
    per = record["per_technique"]
    visible = {
        tech: per[tech]
        for tech in USER_FACING_TECHNIQUES
        if tech in per
    }
    top_visible = max(
        visible.items(),
        key=lambda item: (item[1].get("predicted_frame_fraction") or 0.0, item[1].get("max_probability") or 0.0),
    )
    predicted_visible = [
        tech
        for tech, item in visible.items()
        if (item.get("predicted_frame_fraction") or 0.0) >= 0.05
    ]
    return {
        "top_user_facing_label": top_visible[0],
        "top_user_facing_predicted_fraction": top_visible[1].get("predicted_frame_fraction"),
        "predicted_user_facing_labels_at_5pct_frames": predicted_visible,
        "mix_predicted_fraction": per.get("mix", {}).get("predicted_frame_fraction"),
        "per_technique": {
            tech: {
                "role": item.get("role"),
                "predicted_frame_fraction": item.get("predicted_frame_fraction"),
                "mean_probability": item.get("mean_probability"),
                "max_probability": item.get("max_probability"),
            }
            for tech, item in per.items()
            if tech in TECHNIQUES
        },
        "note": "Model C is frame-level; mix is hidden context, not user-facing.",
    }


def _align_1d(values: np.ndarray, n_frames: int, fill: float = 0.0) -> np.ndarray:
    out = np.full(n_frames, fill, dtype=np.float32)
    n = min(n_frames, len(values))
    if n:
        out[:n] = np.asarray(values[:n], dtype=np.float32)
    return out


def nanopitch_features_from_audio(audio: np.ndarray, n_frames: int, checkpoint: Path, device: str) -> np.ndarray:
    raw = run_nanopitch(audio, checkpoint, device)
    f0 = _align_1d(np.asarray(raw["f0_hz"], dtype=np.float32), n_frames)
    vad_prob = _align_1d(np.asarray(raw["vad_prob"], dtype=np.float32), n_frames)
    voiced = (vad_prob >= 0.5).astype(np.float32)
    conf = _align_1d(np.asarray(raw["pitch_confidence"], dtype=np.float32), n_frames)
    margin = _align_1d(np.asarray(raw.get("pitch_margin", np.zeros_like(f0)), dtype=np.float32), n_frames)
    log_f0 = np.zeros_like(f0, dtype=np.float32)
    valid = f0 > 0
    log_f0[valid] = np.log2(f0[valid] / 32.7).astype(np.float32)
    delta = np.zeros_like(log_f0)
    delta[1:] = log_f0[1:] - log_f0[:-1]
    dropout = (voiced < 0.5).astype(np.float32)
    return np.stack([log_f0, vad_prob, voiced, conf, margin, delta, dropout], axis=0).astype(np.float32)


@torch.no_grad()
def model_c_wav_prediction(
    audio: np.ndarray,
    model_c: torch.nn.Module,
    payload: dict[str, Any],
    device: torch.device,
    nanopitch_checkpoint: Path,
    nanopitch_device: str,
) -> dict[str, Any]:
    hcqt_ext = HCQTExtractor(sr=SR, hop_length=HOP_LENGTH, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    vad_ext = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)
    hcqt = hcqt_ext.compute(audio)
    vad_features = vad_ext.compute(audio)
    n_frames = min(hcqt.shape[2], vad_features.shape[1])
    hcqt = hcqt[:, :, :n_frames].astype(np.float32)
    vad_features = vad_features[:, :n_frames].astype(np.float32)
    nano = nanopitch_features_from_audio(audio, n_frames, nanopitch_checkpoint, nanopitch_device)
    out = model_c(
        torch.from_numpy(hcqt).unsqueeze(0).to(device),
        torch.from_numpy(vad_features).unsqueeze(0).to(device),
        torch.from_numpy(nano).unsqueeze(0).to(device),
    )
    scores = torch.sigmoid(out["technique_logits"]).squeeze(0).cpu().numpy()
    thresholds = payload.get("thresholds") or {tech: 0.5 for tech in TECHNIQUES}
    postprocessing = payload.get("postprocessing") or {}
    min_segment_frames = int(postprocessing.get("min_segment_frames", 20))
    max_gap_frames = int(postprocessing.get("max_gap_frames", 3))
    per_technique: dict[str, Any] = {}
    for tech in TECHNIQUES:
        idx = TECHNIQUE_TO_IDX[tech]
        threshold = float(thresholds.get(tech, 0.5))
        raw_pred = scores[:, idx] >= threshold
        pred = postprocess_technique_prediction(
            scores[:, idx],
            tech,
            threshold,
            min_segment_frames=min_segment_frames,
            max_gap_frames=max_gap_frames,
        )
        per_technique[tech] = {
            "role": "hidden_context" if tech in HIDDEN_CONTEXT_TECHNIQUES else "user_facing_candidate",
            "user_facing_eligible": tech in USER_FACING_TECHNIQUES,
            "threshold": threshold,
            "mean_probability": float(np.mean(scores[:, idx])) if scores.size else 0.0,
            "max_probability": float(np.max(scores[:, idx])) if scores.size else 0.0,
            "raw_predicted_frame_fraction": float(np.mean(raw_pred)) if raw_pred.size else 0.0,
            "predicted_frame_fraction": float(np.mean(pred)) if pred.size else 0.0,
        }
    return summarize_model_c_prediction({"per_technique": per_technique})


def self_wav_predictions(args: argparse.Namespace, model_c: torch.nn.Module, model_c_payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    device = device_for_arg(args.device)
    for path in SELF_WAVS:
        audio = load_audio(path)
        model_a = top_model_a_technique(path, args.model_a_checkpoint, args.device)
        model_c_pred = model_c_wav_prediction(
            audio,
            model_c,
            model_c_payload,
            device,
            args.nanopitch_checkpoint,
            args.device,
        )
        out.append(
            {
                "sample": path.stem,
                "audio_path": str(path),
                "model_a": model_a,
                "model_c": model_c_pred,
                "accuracy_claim": "not_available_no_technique_ground_truth",
            }
        )
    return out


def labelled_clip_comparison(args: argparse.Namespace, model_c: torch.nn.Module, model_c_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_manifest(args.manifest)
    vocab = model_c_payload.get("phoneme_vocab")
    thresholds = model_c_payload.get("thresholds") or {tech: 0.5 for tech in TECHNIQUES}
    device = device_for_arg(args.device)
    out: list[dict[str, Any]] = []
    for split in args.splits:
        for row in stratified_rows(rows, split, args.clips_per_label):
            true_source = row.get("source_technique", "")
            audio_path = Path(row["audio_path"])
            model_a = top_model_a_technique(audio_path, args.model_a_checkpoint, args.device)
            model_c_raw = predict_model_c_clip(model_c, row, vocab, thresholds, device)
            model_c_summary = summarize_model_c_prediction(model_c_raw)
            comparable_label = MODEL_A_TO_MODEL_C_SHARED.get(true_source)
            hidden_context = true_source in HIDDEN_CONTEXT_LABELS
            model_a_top = model_a["top_label"]
            model_c_predicted = set(model_c_summary["predicted_user_facing_labels_at_5pct_frames"])
            out.append(
                {
                    "split": split,
                    "audio_path": str(audio_path),
                    "source_technique": true_source,
                    "comparable_label": comparable_label,
                    "hidden_context_label": hidden_context,
                    "model_a": model_a,
                    "model_c": model_c_summary,
                    "clip_level_scoring": {
                        "model_a_top1_matches_source": model_a_top == true_source,
                        "model_a_top1_matches_comparable_user_label": (
                            comparable_label is not None and model_a_top == comparable_label
                        ),
                        "model_c_predicts_comparable_user_label": (
                            comparable_label is not None and comparable_label in model_c_predicted
                        ),
                        "scoring_note": (
                            "Rough clip-level check only. Model A is clip-level softmax; Model C is frame-level multi-label."
                        ),
                    },
                }
            )
    return out


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({r["split"] for r in records}):
        items = [r for r in records if r["split"] == split]
        comparable = [r for r in items if r["comparable_label"]]
        by_label: dict[str, Any] = {}
        for label in sorted({r["comparable_label"] for r in comparable if r["comparable_label"]}):
            label_items = [r for r in comparable if r["comparable_label"] == label]
            by_label[label] = {
                "clips": len(label_items),
                "model_a_top1_match_rate": _mean([r["clip_level_scoring"]["model_a_top1_matches_comparable_user_label"] for r in label_items]),
                "model_c_clip_detect_rate": _mean([r["clip_level_scoring"]["model_c_predicts_comparable_user_label"] for r in label_items]),
                "model_a_top_counts": dict(Counter(r["model_a"]["top_label"] for r in label_items)),
                "model_c_top_counts": dict(Counter(r["model_c"]["top_user_facing_label"] for r in label_items)),
            }
        by_split[split] = {
            "clips": len(items),
            "comparable_clips": len(comparable),
            "hidden_context_clips": sum(1 for r in items if r["hidden_context_label"]),
            "model_a_top1_match_rate_on_comparable": _mean(
                [r["clip_level_scoring"]["model_a_top1_matches_comparable_user_label"] for r in comparable]
            ),
            "model_c_detect_rate_on_comparable": _mean(
                [r["clip_level_scoring"]["model_c_predicts_comparable_user_label"] for r in comparable]
            ),
            "by_label": by_label,
        }
    return by_split


def _mean(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Model A vs Model C Comparison",
        "",
        "This report compares only outputs that both models can reasonably expose. It does not claim that the models are interchangeable.",
        "",
        "## Model Roles",
        "",
        "- Model A unified: production pitch/VAD/onset/breath-ish heads plus a 20-class clip-level technique head that prior audits marked unreliable.",
        "- Model C v3: NanoPitch-conditioned frame-level six-label technique recognizer. `mix` is hidden context and blocked from user-facing coaching.",
        "- Hybrid stack: Model A + NanoPitch + pyin remains the production-safe pitch/VAD strategy.",
        "",
        "## Self-Recorded WAVs",
        "",
        "The self-recorded WAVs have task expectations but no frame/clip technique labels, so technique accuracy cannot be computed.",
        "",
        "| Sample | Model A top technique | Prob. | Model A voiced | Model C top visible label | Model C frame fraction |",
        "|---|---|---:|---:|---|---:|",
    ]
    for item in payload["self_wav_predictions"]:
        ma = item["model_a"]
        mc = item["model_c"]
        lines.append(
            f"| `{item['sample']}` | `{ma['top_label']}` | {ma['top_probability']:.3f} | {ma['voiced_fraction']:.3f} | `{mc['top_user_facing_label']}` | {mc['top_user_facing_predicted_fraction']:.3f} |"
        )
    lines += [
        "",
        "## Labelled Clip Comparison",
        "",
        "Rough clip-level comparison on bounded manifest clips. For GTsinger/VocalSet labels that overlap both models, this checks whether Model A's top clip label matches the source label and whether Model C predicts the label on at least 5% of frames.",
        "",
        "| Split | Clips | Comparable clips | Hidden context clips | Model A top-1 match | Model C detect rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, summary in payload["aggregate"].items():
        lines.append(
            "| {split} | {clips} | {comp} | {hidden} | {ma} | {mc} |".format(
                split=f"`{split}`",
                clips=summary["clips"],
                comp=summary["comparable_clips"],
                hidden=summary["hidden_context_clips"],
                ma=_fmt(summary["model_a_top1_match_rate_on_comparable"]),
                mc=_fmt(summary["model_c_detect_rate_on_comparable"]),
            )
        )
    lines += ["", "## Per-Label Clip Rates", ""]
    for split, summary in payload["aggregate"].items():
        lines += [
            f"### `{split}`",
            "",
            "| Label | Clips | Model A top-1 match | Model C detect rate | Model A top counts | Model C top counts |",
            "|---|---:|---:|---:|---|---|",
        ]
        for label, item in summary["by_label"].items():
            lines.append(
                "| {label} | {clips} | {ma} | {mc} | `{ma_counts}` | `{mc_counts}` |".format(
                    label=f"`{label}`",
                    clips=item["clips"],
                    ma=_fmt(item["model_a_top1_match_rate"]),
                    mc=_fmt(item["model_c_clip_detect_rate"]),
                    ma_counts=item["model_a_top_counts"],
                    mc_counts=item["model_c_top_counts"],
                )
            )
        lines.append("")
    lines += [
        "## Decision",
        "",
        "- Do not switch from the current hybrid production stack to Model C.",
        "- Model C v3 is not a replacement for Model A because it does not provide production pitch/VAD/task outputs.",
        "- Model C v3 is also not ready as a user-facing technique layer because held-out singer gates still fail.",
        "- Keep Model C report-only until held-out precision and false-positive gates pass per technique.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ml_new/model_c/manifests/model_c.csv"))
    parser.add_argument("--model-a-checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--model-c-checkpoint", type=Path, default=Path("ml_new/checkpoints/model_c/run_alto2_v3/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=NANOPITCH_CHECKPOINT)
    parser.add_argument("--phoneme-vocab", type=Path, default=Path("ml_new/model_c/manifests/phoneme_vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison/model_a_vs_model_c"))
    parser.add_argument("--splits", nargs="*", default=["test_alto1", "test_tenor1", "test_vocalset"])
    parser.add_argument("--clips-per-label", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_for_arg(args.device)
    model_c, model_c_payload = load_model_c(args.model_c_checkpoint, args.phoneme_vocab, device)
    model_c.eval()
    self_preds = self_wav_predictions(args, model_c, model_c_payload)
    labelled = labelled_clip_comparison(args, model_c, model_c_payload)
    payload = {
        "model_a_checkpoint": str(args.model_a_checkpoint),
        "model_c_checkpoint": str(args.model_c_checkpoint),
        "splits": args.splits,
        "clips_per_label": args.clips_per_label,
        "shared_user_facing_labels": list(MODEL_A_TO_MODEL_C_SHARED),
        "hidden_context_labels": list(HIDDEN_CONTEXT_LABELS),
        "self_wav_predictions": self_preds,
        "labelled_clip_records": labelled,
        "aggregate": aggregate(labelled),
        "decision": {
            "switch_to_model_c": False,
            "reason": "Model C does not replace pitch/VAD/task outputs and does not pass held-out technique gates.",
        },
    }
    json_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "SUMMARY.md"
    records_path = args.output_dir / "labelled_clip_records.jsonl"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as fh:
        for record in labelled:
            fh.write(json.dumps(record) + "\n")
    write_markdown(md_path, payload)
    print(json.dumps({"status": "complete", "summary": str(json_path), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
