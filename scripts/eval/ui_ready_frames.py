#!/usr/bin/env python3
"""Export normalized 10 ms selected-source frame data for UI experiments.

This is an eval/debug export only. It does not change backend API responses,
user-facing scoring, model architecture, training, or frontend behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import HOP_LENGTH, HOP_S, SR  # noqa: E402
from scripts.eval import hybrid_decision_harness as h1  # noqa: E402


SELF_RECORDED = h1.SELF_RECORDED


def safe_float(value: Any, precision: int = 6) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return round(val, precision)


TONE_PROXY_CAVEAT = "Spectral/tone metrics are proxy features, not timbre or technique diagnosis."


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def spectral_proxy(frame: np.ndarray) -> dict[str, Any]:
    if frame.size == 0:
        return {
            "is_proxy": True,
            "spectral_centroid_hz": None,
            "spectral_flatness": None,
            "spectral_rolloff_hz": None,
            "low_frequency_ratio": None,
            "harmonicity_noise_proxy": None,
            "noise_proxy": None,
            "caveat": TONE_PROXY_CAVEAT,
        }

    window_size = max(512, int(frame.size))
    padded = np.zeros(window_size, dtype=np.float64)
    padded[: frame.size] = frame.astype(np.float64)
    padded *= np.hanning(window_size)
    spectrum = np.abs(np.fft.rfft(padded)) ** 2
    freqs = np.fft.rfftfreq(window_size, d=1.0 / SR)
    total = float(np.sum(spectrum))
    if total <= 1e-18:
        return {
            "is_proxy": True,
            "spectral_centroid_hz": None,
            "spectral_flatness": None,
            "spectral_rolloff_hz": None,
            "low_frequency_ratio": None,
            "harmonicity_noise_proxy": None,
            "noise_proxy": None,
            "caveat": TONE_PROXY_CAVEAT,
        }

    centroid = float(np.sum(freqs * spectrum) / total)
    positive = spectrum + 1e-18
    flatness = float(np.exp(np.mean(np.log(positive))) / max(np.mean(positive), 1e-18))
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total, side="left"))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    low_ratio = float(np.sum(spectrum[freqs <= 150.0]) / total)
    noise_proxy = clamp01(flatness)
    harmonicity_proxy = clamp01(1.0 - noise_proxy)
    return {
        "is_proxy": True,
        "spectral_centroid_hz": safe_float(centroid),
        "spectral_flatness": safe_float(flatness),
        "spectral_rolloff_hz": safe_float(rolloff),
        "low_frequency_ratio": safe_float(low_ratio),
        "harmonicity_noise_proxy": safe_float(harmonicity_proxy),
        "noise_proxy": safe_float(noise_proxy),
        "caveat": TONE_PROXY_CAVEAT,
    }


def audio_frame_features(audio: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rms = np.zeros(n_frames, dtype=np.float32)
    clipped = np.zeros(n_frames, dtype=bool)
    spectral = []
    for i in range(n_frames):
        start = i * HOP_LENGTH
        end = min(start + HOP_LENGTH, len(audio))
        frame = audio[start:end]
        if frame.size == 0:
            spectral.append(spectral_proxy(frame))
            continue
        rms[i] = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))
        clipped[i] = bool(np.any(np.abs(frame) >= 0.98))
        spectral.append(spectral_proxy(frame))
    return rms, clipped, spectral


def db_from_rms(rms: float) -> float | None:
    if rms <= 0:
        return None
    return float(20.0 * math.log10(max(rms, 1e-12)))


def track_value(track: h1.SourceTrack, i: int) -> dict[str, Any]:
    f0 = float(track.f0_hz[i]) if i < len(track.f0_hz) and np.isfinite(track.f0_hz[i]) and track.f0_hz[i] > 0 else None
    voice_conf = (
        float(track.voice_confidence[i])
        if track.voice_confidence is not None and i < len(track.voice_confidence) and np.isfinite(track.voice_confidence[i])
        else None
    )
    pitch_conf = (
        float(track.pitch_confidence[i])
        if track.pitch_confidence is not None and i < len(track.pitch_confidence) and np.isfinite(track.pitch_confidence[i])
        else None
    )
    return {
        "f0_hz": safe_float(f0),
        "voiced": bool(track.voiced[i]) if i < len(track.voiced) else False,
        "voice_confidence": safe_float(voice_conf),
        "pitch_confidence": safe_float(pitch_conf),
    }


def f0_disagreement_cents(values: list[float]) -> float | None:
    vals = np.asarray([v for v in values if v is not None and v > 0], dtype=np.float64)
    if vals.size < 2:
        return None
    cents = 1200.0 * np.log2(vals / max(float(np.median(vals)), 1e-9))
    return float(np.max(cents) - np.min(cents))


def select_vad(
    recommendation: str,
    tracks: dict[str, h1.SourceTrack],
    i: int,
) -> tuple[bool, float | None, str]:
    model = tracks["model_a"]
    nano = tracks["nanopitch"]
    pyin = tracks["pyin"]

    def conf(track: h1.SourceTrack) -> float | None:
        if track.voice_confidence is None or i >= len(track.voice_confidence):
            return None
        return float(track.voice_confidence[i])

    if recommendation == "nanopitch_guard":
        return bool(nano.voiced[i]), conf(nano), "nanopitch"

    if recommendation == "nanopitch_plus_model_or_pyin":
        support = bool(model.voiced[i] or pyin.voiced[i])
        voiced = bool(nano.voiced[i] and support)
        vals = [v for v in (conf(nano), conf(model), conf(pyin)) if v is not None]
        return voiced, float(np.mean(vals)) if vals else None, "nanopitch_plus_model_or_pyin"

    if recommendation == "model_a_plus_pyin_with_nanopitch_caveat":
        voiced = bool(model.voiced[i] or pyin.voiced[i])
        vals = [v for v in (conf(model), conf(pyin)) if v is not None]
        return voiced, float(max(vals)) if vals else None, "model_a_or_pyin"

    if recommendation == "none":
        return False, None, "none"

    # Conservative fallback: require support from at least one non-NanoPitch
    # source because NanoPitch can be too sparse on real singing.
    voiced = bool(model.voiced[i] or pyin.voiced[i])
    vals = [v for v in (conf(model), conf(pyin), conf(nano)) if v is not None]
    return voiced, float(np.mean(vals)) if vals else None, "hybrid"


def select_f0(
    recommendation: str,
    tracks: dict[str, h1.SourceTrack],
    i: int,
) -> tuple[float | None, float | None, str, list[str]]:
    model = tracks["model_a"]
    nano = tracks["nanopitch"]
    pyin = tracks["pyin"]
    debug_flags: list[str] = []

    def usable(track: h1.SourceTrack) -> bool:
        return (
            i < len(track.f0_hz)
            and bool(track.voiced[i])
            and np.isfinite(track.f0_hz[i])
            and float(track.f0_hz[i]) > 0
        )

    def pitch_conf(track: h1.SourceTrack) -> float | None:
        if track.pitch_confidence is None or i >= len(track.pitch_confidence):
            return None
        val = float(track.pitch_confidence[i])
        return val if math.isfinite(val) else None

    if recommendation == "none":
        return None, None, "none", ["no_selected_f0_source"]

    if recommendation == "pyin":
        if usable(pyin):
            return float(pyin.f0_hz[i]), pitch_conf(pyin), "pyin", debug_flags
        return None, pitch_conf(pyin), "pyin", ["selected_f0_source_unvoiced"]

    if recommendation == "nanopitch":
        if usable(nano):
            return float(nano.f0_hz[i]), pitch_conf(nano), "nanopitch", debug_flags
        return None, pitch_conf(nano), "nanopitch", ["selected_f0_source_unvoiced"]

    if recommendation in {"model_a", "model_a_with_pyin_guard"}:
        if usable(model):
            if usable(pyin):
                cents = abs(1200.0 * math.log2(float(model.f0_hz[i]) / max(float(pyin.f0_hz[i]), 1e-9)))
                if cents > 200.0:
                    debug_flags.append("model_a_pyin_f0_disagreement_gt_200_cents")
            elif recommendation == "model_a_with_pyin_guard":
                debug_flags.append("pyin_guard_unavailable")
            return float(model.f0_hz[i]), pitch_conf(model), "model_a", debug_flags
        if recommendation == "model_a_with_pyin_guard" and usable(pyin):
            debug_flags.append("model_a_unvoiced_pyin_fallback")
            return float(pyin.f0_hz[i]), pitch_conf(pyin), "pyin", debug_flags
        return None, pitch_conf(model), "model_a", ["selected_f0_source_unvoiced"]

    # Unknown/hybrid recommendation: prefer pyin if stable/voiced, then model A,
    # then NanoPitch.
    for name, track in (("pyin", pyin), ("model_a", model), ("nanopitch", nano)):
        if usable(track):
            return float(track.f0_hz[i]), pitch_conf(track), name, ["hybrid_f0_fallback"]
    return None, None, "none", ["no_selected_f0_source"]


def build_ui_ready_frames(
    audio_path: Path,
    *,
    checkpoint: Path = Path("ml_new/checkpoints/unified/best.pt"),
    nanopitch_checkpoint: Path = h1.NANOPITCH_CHECKPOINT,
    device: str = "cpu",
    debug: bool = True,
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audio_path = audio_path.resolve()
    audio = h1.load_audio(audio_path)
    tracks_list = h1.align_tracks(
        [
            h1.load_model_a_track(audio_path, audio, checkpoint, device),
            h1.load_nanopitch_track(audio_path, audio, nanopitch_checkpoint, device),
            h1.load_pyin_track(audio),
        ]
    )
    tracks = {track.name: track for track in tracks_list}
    metrics = h1.compute_metrics(audio_path, tracks_list, task_config=task_config)
    rec = metrics["recommendation"]
    n_frames = metrics["frames"]
    rms, clipped, spectral = audio_frame_features(audio, n_frames)
    selected_vad_rec = rec["selected_vad_source_recommendation"]
    selected_f0_rec = rec["selected_f0_source_recommendation"]

    frames = []
    for i in range(n_frames):
        selected_voiced, voice_conf, selected_vad_source = select_vad(selected_vad_rec, tracks, i)
        f0_hz, pitch_conf, selected_f0_source, f0_flags = select_f0(selected_f0_rec, tracks, i)

        source_values = {name: track_value(track, i) for name, track in tracks.items()}
        source_f0s = [
            val["f0_hz"]
            for val in source_values.values()
            if val["voiced"] and val["f0_hz"] is not None
        ]
        cents_spread = f0_disagreement_cents(source_f0s)
        source_disagreement = cents_spread is not None and cents_spread > 200.0
        near_silence = bool(rms[i] < 1e-4)
        low_confidence = bool(
            selected_voiced
            and (
                (voice_conf is not None and voice_conf < 0.3)
                or (pitch_conf is not None and pitch_conf < 0.2)
                or source_disagreement
            )
        )
        noisy = bool(
            source_values["nanopitch"]["voiced"] is False
            and (source_values["model_a"]["voiced"] or source_values["pyin"]["voiced"])
            and selected_vad_rec == "nanopitch_guard"
        )
        caveats: list[str] = []
        debug_flags = list(f0_flags)
        if source_disagreement:
            debug_flags.append("source_f0_disagreement_gt_200_cents")
            caveats.append("Pitch sources disagree on this frame.")
        if noisy:
            debug_flags.append("nanopitch_negative_other_sources_positive")
        if low_confidence:
            debug_flags.append("low_confidence_selected_frame")
        if selected_f0_rec == "none":
            caveats.append("No f0 source selected for this frame.")

        frame = {
            "time_s": round(i * HOP_S, 6),
            "frame_index": i,
            "f0_hz": safe_float(f0_hz),
            "voiced": bool(selected_voiced),
            "voice_confidence": safe_float(voice_conf),
            "pitch_confidence": safe_float(pitch_conf),
            "selected_f0_source": selected_f0_source,
            "selected_vad_source": selected_vad_source,
            "volume": {
                "rms": safe_float(float(rms[i])),
                "rms_db": safe_float(db_from_rms(float(rms[i]))),
            },
            "spectral_tone_proxy": spectral[i],
            "signal_quality": {
                "clipped": bool(clipped[i]),
                "near_silence": near_silence,
                "noisy": noisy,
                "low_confidence": low_confidence,
                "source_disagreement": source_disagreement,
            },
            "caveats": caveats,
            "debug_flags": sorted(set(debug_flags)),
        }
        if debug:
            frame["source_values"] = source_values
        frames.append(frame)

    return {
        "schema_version": "h1b.ui_ready_frames.v1",
        "input_path": str(audio_path),
        "audio": {
            "duration_s": safe_float(len(audio) / SR),
            "sample_rate": SR,
            "hop_s": HOP_S,
            "channels": 1,
        },
        "task_config": task_config
        or {
            "task_type": rec["task_type"],
            "source": "inferred_from_h1_harness",
        },
        "source_strategy": rec,
        "hybrid_metrics": {
            "voiced_agreement": metrics["voiced_agreement"],
            "f0_disagreement": metrics["f0_disagreement"],
        },
        "debug": {
            "source_values_included": debug,
            "source_summaries": metrics["sources"],
        },
        "frames": frames,
    }


def first_interesting_frame(export: dict[str, Any]) -> dict[str, Any] | None:
    for frame in export["frames"]:
        if frame["f0_hz"] is not None and not frame["signal_quality"].get("near_silence"):
            return frame
    for frame in export["frames"]:
        if frame["voiced"] and not frame["signal_quality"].get("near_silence"):
            return frame
    for frame in export["frames"]:
        if frame["debug_flags"] and not frame["signal_quality"].get("near_silence"):
            return frame
    for frame in export["frames"]:
        if frame["voiced"] or frame["f0_hz"] is not None or frame["debug_flags"]:
            return frame
    return export["frames"][0] if export["frames"] else None


def compact_frame_example(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    keys = [
        "time_s",
        "frame_index",
        "f0_hz",
        "voiced",
        "voice_confidence",
        "pitch_confidence",
        "selected_f0_source",
        "selected_vad_source",
        "volume",
        "spectral_tone_proxy",
        "signal_quality",
        "caveats",
        "debug_flags",
    ]
    return {key: frame.get(key) for key in keys}


def write_report(path: Path, exports: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# H1b UI-Ready Frames Report",
        "",
        "Report-only normalized selected-source frame export for the five self-recorded WAV samples.",
        "",
        "No frontend, model architecture, training, user-facing scoring, or existing API behavior was changed.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        "- Per-sample JSON files are saved as `reports/ui_ready_frames/<sample>/<sample>_ui_ready_frames.json`.",
        "",
        "## Summary",
        "",
        "| Sample | Task type | F0 recommendation | VAD recommendation | Frames | Voiced % | Selected f0 frames | Main caveat |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for export in exports:
        sample = Path(export["input_path"]).stem
        rec = export["source_strategy"]
        frames = export["frames"]
        voiced = sum(1 for frame in frames if frame["voiced"])
        f0_count = sum(1 for frame in frames if frame["f0_hz"] is not None)
        caveat = ""
        for frame in frames:
            if frame["caveats"]:
                caveat = frame["caveats"][0]
                break
        lines.append(
            "| {sample} | `{task}` | `{f0}` | `{vad}` | {frames_n} | {voiced_pct:.1f}% | {f0_count} | {caveat} |".format(
                sample=f"`{sample}`",
                task=rec["task_type"],
                f0=rec["selected_f0_source_recommendation"],
                vad=rec["selected_vad_source_recommendation"],
                frames_n=len(frames),
                voiced_pct=100.0 * voiced / max(len(frames), 1),
                f0_count=f0_count,
                caveat=caveat.replace("|", "\\|"),
            )
        )

    lines += [
        "",
        "## Frame Examples By Task Type",
        "",
        "Examples are compacted to the UI-relevant fields; full debug `source_values` are in the JSON artifacts.",
        "",
    ]
    seen: set[str] = set()
    for export in exports:
        rec = export["source_strategy"]
        task = rec["task_type"]
        if task in seen:
            continue
        seen.add(task)
        sample = Path(export["input_path"]).stem
        example = compact_frame_example(first_interesting_frame(export))
        lines += [
            f"### `{task}` example: `{sample}`",
            "",
            "```json",
            json.dumps(example, indent=2),
            "```",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- `selected_f0_source` is per-frame actual source after applying the H1 recommendation.",
        "- `selected_vad_source` can be a concrete source or a small hybrid rule label such as `model_a_or_pyin`.",
        "- `source_values` are included only because this eval export runs in debug mode.",
        "- `target_f0_hz`, `cents_error`, and segment-level regions are intentionally out of scope for H1b.",
        "- This is the smallest UI-facing frame slice from H0, not the full H0 contract.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ui_ready_frames"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=h1.NANOPITCH_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = args.files if args.files else SELF_RECORDED
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Exporting UI-ready frames for {path}")
        export = build_ui_ready_frames(
            path,
            checkpoint=args.checkpoint,
            nanopitch_checkpoint=args.nanopitch_checkpoint,
            device=args.device,
            debug=not args.no_debug,
        )
        sample = path.stem
        sample_dir = args.output_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"{sample}_ui_ready_frames.json"
        out_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
        exports.append(export)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(exports, indent=2), encoding="utf-8")
    write_report(Path("H1B_UI_READY_FRAMES_REPORT.md"), exports, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary_path),
                "report": "H1B_UI_READY_FRAMES_REPORT.md",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
