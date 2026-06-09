#!/usr/bin/env python3
"""Evaluate Model A, NanoPitch, pyin, and hybrid behavior on annotated singing manifests.

This is report-only. It does not retrain models, change checkpoints, or update
product scoring. The first intended target is MIR-1K frame-level VAD/f0
evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import HOP_S, SR  # noqa: E402
from scripts.eval.evaluate_nanopitch_wav import DEFAULT_CHECKPOINT as NANOPITCH_CHECKPOINT  # noqa: E402
from scripts.eval.hybrid_decision_harness import (  # noqa: E402
    SourceTrack,
    align_tracks,
    compute_metrics,
    load_audio,
    load_model_a_track,
    load_nanopitch_track,
    load_pyin_track,
)


SOURCE_ORDER = ("model_a", "nanopitch", "pyin", "hybrid_majority")


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Create it first, for example:\n"
            "  ml/.venv/bin/python scripts/data/prepare_human_singing_manifest.py "
            "--dataset mir1k --root /path/to/MIR-1K --license-acknowledged "
            "--output data/manifests/human_singing/mir1k.jsonl"
        )
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
        if limit is not None and len(records) >= limit:
            break
    return records


def require_ground_truth(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ann = record.get("frame_annotations") or {}
    status = ann.get("status")
    if status == "not_loaded_by_manifest_builder" or ann.get("f0_hz") is None or ann.get("voiced") is None:
        raise ValueError(
            f"{record.get('audio_path')}: missing ground-truth frame annotations. "
            "Dataset-specific adapter must load f0_hz and voiced before evaluation."
        )
    times = np.asarray(ann.get("time_s") or [], dtype=np.float64)
    f0 = np.asarray(ann.get("f0_hz") or [], dtype=np.float64)
    voiced = np.asarray(ann.get("voiced") or [], dtype=bool)
    n = min(len(times), len(f0), len(voiced))
    if n == 0:
        raise ValueError(f"{record.get('audio_path')}: empty ground-truth frame annotations.")
    return times[:n], f0[:n], voiced[:n]


def align_track_to_reference(track: SourceTrack, ref_times: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.rint(ref_times / HOP_S).astype(int)
    idx = np.clip(idx, 0, max(len(track.f0_hz) - 1, 0))
    f0 = np.asarray(track.f0_hz, dtype=np.float64)[idx]
    voiced = np.asarray(track.voiced, dtype=bool)[idx]
    voice_conf = (
        np.asarray(track.voice_confidence, dtype=np.float64)[idx]
        if track.voice_confidence is not None
        else np.full(len(ref_times), np.nan)
    )
    pitch_conf = (
        np.asarray(track.pitch_confidence, dtype=np.float64)[idx]
        if track.pitch_confidence is not None
        else np.full(len(ref_times), np.nan)
    )
    return {"f0_hz": f0, "voiced": voiced, "voice_confidence": voice_conf, "pitch_confidence": pitch_conf}


def hybrid_majority_track(aligned_sources: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    voices = np.stack([aligned_sources[name]["voiced"] for name in ("model_a", "nanopitch", "pyin")], axis=0)
    voiced = np.sum(voices, axis=0) >= 2
    f0 = np.zeros(voiced.shape, dtype=np.float64)
    pitch_conf = np.full(voiced.shape, np.nan, dtype=np.float64)
    voice_conf = np.mean(
        np.stack([np.nan_to_num(aligned_sources[name]["voice_confidence"], nan=0.0) for name in ("model_a", "nanopitch", "pyin")], axis=0),
        axis=0,
    )
    for i in range(len(voiced)):
        vals = [
            float(aligned_sources[name]["f0_hz"][i])
            for name in ("model_a", "nanopitch", "pyin")
            if aligned_sources[name]["voiced"][i] and aligned_sources[name]["f0_hz"][i] > 0
        ]
        if voiced[i] and vals:
            f0[i] = float(np.median(vals))
            pitch_conf[i] = float(np.mean([1.0 / max(len(vals), 1)]))
    return {"f0_hz": f0, "voiced": voiced, "voice_confidence": voice_conf, "pitch_confidence": pitch_conf}


def cents_error(ref_f0: np.ndarray, est_f0: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_f0, dtype=np.float64)
    est = np.asarray(est_f0, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(est) & (ref > 0) & (est > 0)
    out = np.full(ref.shape, np.nan, dtype=np.float64)
    out[valid] = 1200.0 * np.log2(est[valid] / ref[valid])
    return out


def evaluate_source(ref_f0: np.ndarray, ref_voiced: np.ndarray, est: dict[str, np.ndarray]) -> dict[str, Any]:
    est_voiced = np.asarray(est["voiced"], dtype=bool)
    est_f0 = np.asarray(est["f0_hz"], dtype=np.float64)
    tp = int(np.sum(ref_voiced & est_voiced))
    fp = int(np.sum(~ref_voiced & est_voiced))
    fn = int(np.sum(ref_voiced & ~est_voiced))
    tn = int(np.sum(~ref_voiced & ~est_voiced))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    errors = np.abs(cents_error(ref_f0, est_f0))
    voiced_errors = errors[ref_voiced & est_voiced & np.isfinite(errors)]
    jumps = jump_counts(est_f0)
    return {
        "frames": int(len(ref_voiced)),
        "voiced_precision": float(precision),
        "voiced_recall": float(recall),
        "voiced_f1": float(f1),
        "false_voiced_rate": float(fp / max(fp + tn, 1)),
        "f0_coverage_on_ref_voiced": float(tp / max(int(np.sum(ref_voiced)), 1)),
        "median_abs_f0_error_cents": float(np.median(voiced_errors)) if voiced_errors.size else None,
        "mean_abs_f0_error_cents": float(np.mean(voiced_errors)) if voiced_errors.size else None,
        "octave_error_rate": float(np.mean(voiced_errors >= 900.0)) if voiced_errors.size else None,
        "semitone_error_rate": float(np.mean(voiced_errors >= 100.0)) if voiced_errors.size else None,
        "octave_jump_count": jumps["octave_jump_count"],
        "semitone_jump_count": jumps["semitone_jump_count"],
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def jump_counts(f0_hz: np.ndarray) -> dict[str, int]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    octave = 0
    semitone = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        cents = abs(1200.0 * math.log2(float(f0[b]) / max(float(f0[a]), 1e-9)))
        if cents >= 900.0:
            octave += 1
        if cents >= 200.0:
            semitone += 1
    return {"octave_jump_count": int(octave), "semitone_jump_count": int(semitone)}


def aggregate_source_metrics(per_file: list[dict[str, Any]], source: str) -> dict[str, Any]:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    errors: list[float] = []
    frames = 0
    octave_jumps = 0
    semitone_jumps = 0
    for item in per_file:
        metrics = item["sources"][source]
        frames += int(metrics["frames"])
        for key in conf:
            conf[key] += int(metrics["confusion"][key])
        octave_jumps += int(metrics["octave_jump_count"])
        semitone_jumps += int(metrics["semitone_jump_count"])
        errors.extend(item["f0_abs_error_cents"].get(source, []))
    precision = conf["tp"] / max(conf["tp"] + conf["fp"], 1)
    recall = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    err = np.asarray(errors, dtype=np.float64)
    return {
        "frames": frames,
        "voiced_precision": float(precision),
        "voiced_recall": float(recall),
        "voiced_f1": float(f1),
        "false_voiced_rate": float(conf["fp"] / max(conf["fp"] + conf["tn"], 1)),
        "f0_coverage_on_ref_voiced": float(conf["tp"] / max(conf["tp"] + conf["fn"], 1)),
        "median_abs_f0_error_cents": float(np.median(err)) if err.size else None,
        "mean_abs_f0_error_cents": float(np.mean(err)) if err.size else None,
        "octave_error_rate": float(np.mean(err >= 900.0)) if err.size else None,
        "semitone_error_rate": float(np.mean(err >= 100.0)) if err.size else None,
        "octave_jump_count": octave_jumps,
        "semitone_jump_count": semitone_jumps,
        "confusion": conf,
    }


def process_record(
    record: dict[str, Any],
    output_dir: Path,
    checkpoint: Path,
    nanopitch_checkpoint: Path,
    device: str,
    write_detail: bool,
) -> dict[str, Any]:
    audio_path = Path(record["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found from manifest: {audio_path}")
    ref_times, ref_f0, ref_voiced = require_ground_truth(record)

    started = time.perf_counter()
    audio = load_audio(audio_path)
    tracks = align_tracks(
        [
            load_model_a_track(audio_path, audio, checkpoint, device),
            load_nanopitch_track(audio_path, audio, nanopitch_checkpoint, device),
            load_pyin_track(audio),
        ]
    )
    runtime_s = time.perf_counter() - started

    aligned = {track.name: align_track_to_reference(track, ref_times) for track in tracks}
    aligned["hybrid_majority"] = hybrid_majority_track(aligned)
    source_metrics = {name: evaluate_source(ref_f0, ref_voiced, aligned[name]) for name in SOURCE_ORDER}
    errors = {
        name: [
            round(float(value), 3)
            for value in np.abs(cents_error(ref_f0, aligned[name]["f0_hz"]))[
                ref_voiced & aligned[name]["voiced"] & np.isfinite(cents_error(ref_f0, aligned[name]["f0_hz"]))
            ]
        ]
        for name in SOURCE_ORDER
    }
    hybrid_metrics = compute_metrics(audio_path, tracks)

    result = {
        "dataset": record.get("dataset"),
        "sample": audio_path.stem,
        "input_path": str(audio_path),
        "duration_s": float(len(audio) / SR),
        "reference_frames": int(len(ref_times)),
        "reference_voiced_percentage": float(np.mean(ref_voiced)),
        "annotation_status": (record.get("frame_annotations") or {}).get("status"),
        "runtime_s": float(runtime_s),
        "realtime_factor": float(runtime_s / max(len(audio) / SR, 1e-9)),
        "sources": source_metrics,
        "hybrid_recommendation": hybrid_metrics.get("recommendation"),
        "source_agreement": {
            "voiced_agreement": hybrid_metrics.get("voiced_agreement"),
            "f0_disagreement": hybrid_metrics.get("f0_disagreement"),
        },
        "f0_abs_error_cents": errors,
        "artifacts": {},
    }

    if write_detail:
        sample_dir = output_dir / audio_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        json_path = sample_dir / f"{audio_path.stem}_human_singing_eval.json"
        svg_path = sample_dir / f"{audio_path.stem}_human_singing_eval.svg"
        result["artifacts"] = {"json": str(json_path), "plot": str(svg_path)}
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_svg(svg_path, audio, ref_times, ref_f0, ref_voiced, aligned, audio_path.stem)
    return result


def write_svg(
    path: Path,
    audio: np.ndarray,
    ref_times: np.ndarray,
    ref_f0: np.ndarray,
    ref_voiced: np.ndarray,
    aligned: dict[str, dict[str, np.ndarray]],
    title: str,
) -> None:
    width, height = 1200, 760
    left, right = 82, 32
    plot_w = width - left - right
    duration = max(len(audio) / SR, float(ref_times[-1]) if len(ref_times) else HOP_S, HOP_S)
    rows = [("Waveform", 52, 145), ("Ground truth vs estimates", 195, 455), ("Voiced masks", 520, 710)]
    colors = {"reference": "#111827", "model_a": "#ef4444", "nanopitch": "#7c3aed", "pyin": "#059669", "hybrid_majority": "#2563eb"}

    def x_at(t: float) -> float:
        return left + (t / duration) * plot_w

    valid_f0 = [float(v) for v in ref_f0 if np.isfinite(v) and v > 0]
    for source in SOURCE_ORDER:
        valid_f0.extend(float(v) for v in aligned[source]["f0_hz"] if np.isfinite(v) and v > 0)
    fmin = max(40.0, float(np.percentile(valid_f0, 1)) * 0.8) if valid_f0 else 50.0
    fmax = min(2200.0, float(np.percentile(valid_f0, 99)) * 1.2) if valid_f0 else 800.0
    if fmax <= fmin:
        fmin, fmax = 50.0, 800.0

    def y_f0(hz: float, top: float, bottom: float) -> float:
        frac = (math.log2(max(hz, 1e-9)) - math.log2(fmin)) / max(math.log2(fmax) - math.log2(fmin), 1e-9)
        return bottom - float(np.clip(frac, 0.0, 1.0)) * (bottom - top)

    def poly(points: list[tuple[float, float]], color: str, width_px: float) -> str:
        if not points:
            return ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" />'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">Human singing eval: {escape(title)}</text>',
    ]
    for label, top, bottom in rows:
        svg.append(f'<text x="18" y="{top + 18}" font-size="13" font-family="Arial">{escape(label)}</text>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="#ddd" />')

    if audio.size:
        top, bottom = rows[0][1], rows[0][2]
        bins = min(900, max(1, audio.size // 120))
        step = max(1, audio.size // bins)
        chunks = audio[: bins * step].reshape(bins, step)
        peak = np.max(np.abs(chunks), axis=1)
        max_peak = max(float(np.max(peak)), 1e-9)
        mid = (top + bottom) / 2
        scale = (bottom - top) / 2
        pts_top, pts_bottom = [], []
        for i, val in enumerate(peak):
            x = x_at((i * step) / SR)
            amp = float(val) / max_peak * scale
            pts_top.append((x, mid - amp))
            pts_bottom.append((x, mid + amp))
        area = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts_top + list(reversed(pts_bottom)))
        svg.append(f'<polygon points="{area}" fill="#dbeafe" stroke="#60a5fa" stroke-width="1" />')

    top, bottom = rows[1][1], rows[1][2]
    ref_points = [(x_at(float(t)), y_f0(float(f0), top, bottom)) for t, f0, v in zip(ref_times, ref_f0, ref_voiced) if v and f0 > 0]
    svg.append(poly(ref_points, colors["reference"], 2.6))
    for source in SOURCE_ORDER:
        points = [
            (x_at(float(t)), y_f0(float(f0), top, bottom))
            for t, f0, v in zip(ref_times, aligned[source]["f0_hz"], aligned[source]["voiced"])
            if v and np.isfinite(f0) and f0 > 0
        ]
        svg.append(poly(points, colors[source], 1.5))
    svg.append(f'<text x="{left}" y="{top - 8}" font-size="12" font-family="Arial" fill="#555">{fmax:.1f} Hz</text>')
    svg.append(f'<text x="{left}" y="{bottom + 16}" font-size="12" font-family="Arial" fill="#555">{fmin:.1f} Hz</text>')

    top, _bottom = rows[2][1], rows[2][2]
    lane_h = 24
    masks = {"reference": ref_voiced, **{name: aligned[name]["voiced"] for name in SOURCE_ORDER}}
    for i, (name, mask) in enumerate(masks.items()):
        y = top + 18 + i * 34
        svg.append(f'<text x="{left}" y="{y - 5}" font-size="12" font-family="Arial">{escape(name)}</text>')
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            if start is not None and ((not value) or idx == len(mask) - 1):
                end = idx if not value else idx + 1
                x1 = x_at(float(ref_times[start]))
                x2 = x_at(float(ref_times[min(end - 1, len(ref_times) - 1)] + HOP_S))
                svg.append(f'<rect x="{x1:.2f}" y="{y}" width="{max(x2-x1, 1):.2f}" height="{lane_h}" fill="{colors.get(name, "#999")}" opacity="0.45" />')
                start = None

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_summary_markdown(path: Path, manifest: Path, output_dir: Path, results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    lines = [
        "# Human Singing Dataset Evaluation",
        "",
        f"- Manifest: `{manifest}`",
        f"- Output directory: `{output_dir}`",
        f"- Records evaluated: `{len(results)}`",
        "- Scope: report-only VAD/f0 reliability. No reference-song scoring claims are made.",
        "",
        "## Aggregate Metrics",
        "",
        "| Source | Voiced F1 | False voiced | F0 coverage | Median abs f0 error | Mean abs f0 error | Octave error | Semitone error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_ORDER:
        m = aggregate["sources"][source]
        lines.append(
            "| {source} | {f1:.1f}% | {false:.1f}% | {coverage:.1f}% | {median} | {mean} | {octave} | {semi} |".format(
                source=source,
                f1=100.0 * m["voiced_f1"],
                false=100.0 * m["false_voiced_rate"],
                coverage=100.0 * m["f0_coverage_on_ref_voiced"],
                median=fmt_cents(m["median_abs_f0_error_cents"]),
                mean=fmt_cents(m["mean_abs_f0_error_cents"]),
                octave=fmt_pct(m["octave_error_rate"]),
                semi=fmt_pct(m["semitone_error_rate"]),
            )
        )
    lines += [
        "",
        "## Current Recommendation Rule",
        "",
        "- Prefer the source with the best validated f0 error only when voiced F1 and false-voiced rate are acceptable.",
        "- If Model A has materially worse false-voiced or f0 error than NanoPitch/pyin, keep hybrid guards.",
        "- This report is evidence for source selection and calibration, not user-facing score calibration.",
        "",
        "## Caveats",
        "",
        "- MIR-1K is useful for VAD/f0, not for breath/timbre diagnosis.",
        "- Lyrics/reference-song scoring remains out of scope unless note timing/reference alignment is validated separately.",
        "- Dataset license and citation requirements must be reviewed before any product or training use.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_cents(value: Any) -> str:
    return "null" if value is None else f"{float(value):.1f} cents"


def fmt_pct(value: Any) -> str:
    return "null" if value is None else f"{100.0 * float(value):.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/human_singing/mir1k.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/human_singing_eval/mir1k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=NANOPITCH_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--detail-limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = load_manifest(args.manifest, args.limit)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not records:
        raise SystemExit(f"No records found in manifest: {args.manifest}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Model A checkpoint not found: {args.checkpoint}")
    if not args.nanopitch_checkpoint.exists():
        raise SystemExit(f"NanoPitch checkpoint not found: {args.nanopitch_checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, record in enumerate(records):
        try:
            print(f"Evaluating {idx + 1}/{len(records)} {record.get('audio_path')}")
            results.append(
                process_record(
                    record,
                    args.output_dir,
                    args.checkpoint,
                    args.nanopitch_checkpoint,
                    args.device,
                    write_detail=idx < args.detail_limit,
                )
            )
        except Exception as exc:
            errors.append({"audio_path": str(record.get("audio_path")), "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {record.get('audio_path')}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not results:
        raise SystemExit({"status": "failed", "errors": errors[:10]})

    aggregate = {
        "schema_version": "vocalstars.human_singing_eval.v1",
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "records_requested": len(records),
        "records_evaluated": len(results),
        "errors": errors,
        "sources": {source: aggregate_source_metrics(results, source) for source in SOURCE_ORDER},
    }
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "SUMMARY.md"
    summary_json.write_text(json.dumps({"aggregate": aggregate, "files": results}, indent=2), encoding="utf-8")
    write_summary_markdown(summary_md, args.manifest, args.output_dir, results, aggregate)
    print(json.dumps({"status": "complete", "summary_json": str(summary_json), "summary_md": str(summary_md), "errors": len(errors)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
