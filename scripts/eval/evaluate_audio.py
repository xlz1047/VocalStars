#!/usr/bin/env python3
"""Evaluate one audio file with the existing VocalStars ml_new inference API."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import analyse_recording  # noqa: E402

SR = 16_000
SUPPORTED_DIRECT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".webm"}
CONVERT_EXTS = {".m4a", ".aac", ".mp4"}


def clean_json(obj: Any) -> Any:
    """Convert dataclasses/numpy values to JSON-safe containers."""
    if is_dataclass(obj):
        return clean_json(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj


def ensure_decodable_audio(audio_path: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return an audio path librosa can read, converting .m4a via afconvert if needed."""
    ext = audio_path.suffix.lower()
    info: dict[str, Any] = {
        "input_path": str(audio_path),
        "input_extension": ext,
        "converted": False,
        "conversion_tool": None,
        "conversion_error": None,
        "used_audio_path": str(audio_path),
    }

    if ext not in CONVERT_EXTS:
        return audio_path, info

    converted_dir = output_dir / "_converted_wav"
    converted_dir.mkdir(parents=True, exist_ok=True)
    wav_path = converted_dir / f"{audio_path.stem}.wav"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(SR),
            str(wav_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            info.update(
                {
                    "converted": True,
                    "conversion_tool": "ffmpeg",
                    "conversion_command": " ".join(cmd),
                    "used_audio_path": str(wav_path),
                }
            )
            return wav_path, info
        except subprocess.CalledProcessError as exc:
            info["conversion_tool"] = "ffmpeg"
            info["conversion_command"] = " ".join(cmd)
            info["conversion_error"] = (exc.stderr or exc.stdout or str(exc)).strip()

    afconvert = shutil.which("afconvert")
    if afconvert is None:
        info["conversion_error"] = (
            "M4A input requires conversion, but neither ffmpeg nor afconvert "
            "was found. Convert manually to WAV."
        )
        return audio_path, info

    cmd = [
        afconvert,
        "-f",
        "WAVE",
        "-d",
        "LEI16@16000",
        str(audio_path),
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        info.update(
            {
                "converted": True,
                "conversion_tool": "afconvert",
                "conversion_command": " ".join(cmd),
                "used_audio_path": str(wav_path),
            }
        )
        return wav_path, info
    except subprocess.CalledProcessError as exc:
        info["conversion_tool"] = "afconvert"
        info["conversion_command"] = " ".join(cmd)
        info["conversion_error"] = (exc.stderr or exc.stdout or str(exc)).strip()
        return audio_path, info


def load_waveform(audio_path: Path) -> tuple[np.ndarray, int, str | None]:
    """Load audio for waveform plotting."""
    import librosa

    try:
        audio, sr = librosa.load(str(audio_path), sr=SR, mono=True)
        return audio.astype(np.float32), sr, None
    except Exception as exc:  # pragma: no cover - reported in artifacts
        return np.zeros(0, dtype=np.float32), SR, f"{type(exc).__name__}: {exc}"


def summarize_result(result: Any, audio: np.ndarray, sr: int) -> dict[str, Any]:
    pitch = np.asarray(result.pitch_hz, dtype=np.float32)
    voiced = np.asarray(result.voiced, dtype=bool)
    breath = np.asarray(result.breath_frames, dtype=bool)
    onset = np.asarray(result.onset_frames, dtype=bool)
    voiced_pitch = pitch[(pitch > 0) & voiced]
    duration_s = float(len(audio) / sr) if sr > 0 else float(len(pitch) * result.hop_s)

    summary = {
        "duration_s": round(duration_s, 3),
        "n_frames": int(len(pitch)),
        "hop_s": float(result.hop_s),
        "audio_rms": float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0,
        "voiced_frame_ratio": float(voiced.mean()) if len(voiced) else 0.0,
        "voiced_duration_s": float(voiced.sum() * result.hop_s),
        "mean_f0_hz": float(voiced_pitch.mean()) if len(voiced_pitch) else None,
        "median_f0_hz": float(np.median(voiced_pitch)) if len(voiced_pitch) else None,
        "min_f0_hz": float(voiced_pitch.min()) if len(voiced_pitch) else None,
        "max_f0_hz": float(voiced_pitch.max()) if len(voiced_pitch) else None,
        "pitch_accuracy": float(result.pitch_accuracy),
        "pitch_drift_cents": float(result.pitch_drift_cents),
        "phrase_lengths_s": [float(x) for x in result.phrase_lengths_s],
        "breath_count": int(result.breath_count),
        "breath_frame_ratio": float(breath.mean()) if len(breath) else 0.0,
        "onset_count": int(result.onset_count),
        "onset_frame_ratio": float(onset.mean()) if len(onset) else 0.0,
        "onset_clarity": float(result.onset_clarity),
        "score": int(result.score),
        "technique": str(result.technique),
        "technique_confidence": float(result.technique_confidence),
        "note_count": int(len(result.notes)),
        "voice_quality_available": result.voice_quality is not None,
        "confidence_curve_available": False,
        "confidence_curve_note": (
            "analyse_recording() returns thresholded voiced/breath/onset arrays, "
            "but not raw per-frame confidence probabilities."
        ),
    }
    return summary


def scale(values: np.ndarray, vmin: float, vmax: float, top: float, height: float) -> np.ndarray:
    if vmax <= vmin:
        return np.full_like(values, top + height / 2, dtype=np.float64)
    return top + height - (values - vmin) / (vmax - vmin) * height


def polyline(points: list[tuple[float, float]], color: str, width: float = 1.5) -> str:
    if not points:
        return ""
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="{width}" />'


def make_plot_svg(
    audio: np.ndarray,
    sr: int,
    result: Any,
    title: str,
    out_path: Path,
) -> None:
    """Create a dependency-light SVG plot with waveform, voiced timeline, and f0."""
    width = 1100
    height = 700
    margin_l = 70
    margin_r = 30
    plot_w = width - margin_l - margin_r
    audio_duration = len(audio) / sr if sr and len(audio) else 0.0
    frame_duration = len(result.pitch_hz) * result.hop_s
    duration = max(audio_duration, frame_duration, result.hop_s)

    def x_at(t: np.ndarray | float) -> np.ndarray | float:
        return margin_l + (np.asarray(t) / duration) * plot_w

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{margin_l}" y="34" font-family="Arial" font-size="22" font-weight="700">{escape_xml(title)}</text>',
    ]

    panels = [
        ("Waveform", 70, 140),
        ("Voiced / Breath / Onset", 255, 120),
        ("F0 / Pitch curve", 425, 210),
    ]
    for label, top, ph in panels:
        svg.append(f'<text x="20" y="{top + 18}" font-family="Arial" font-size="13" fill="#333">{label}</text>')
        svg.append(f'<rect x="{margin_l}" y="{top}" width="{plot_w}" height="{ph}" fill="#fafafa" stroke="#ddd" />')

    # Waveform.
    if len(audio):
        n = min(1600, len(audio))
        idx = np.linspace(0, len(audio) - 1, n).astype(int)
        ts = idx / sr
        vals = audio[idx]
        amp = max(float(np.max(np.abs(vals))), 1e-5)
        y = scale(vals, -amp, amp, 70, 140)
        pts = [(float(x_at(t)), float(yy)) for t, yy in zip(ts, y)]
        svg.append(polyline(pts, "#315efb", 1.0))
        y0 = scale(np.array([0.0]), -amp, amp, 70, 140)[0]
        svg.append(f'<line x1="{margin_l}" y1="{y0:.2f}" x2="{margin_l + plot_w}" y2="{y0:.2f}" stroke="#bbb" stroke-width="1" />')

    # Timelines.
    timelines = [
        ("voiced", np.asarray(result.voiced, dtype=bool), 280, "#16803c"),
        ("breath", np.asarray(result.breath_frames, dtype=bool), 315, "#d97706"),
        ("onset", np.asarray(result.onset_frames, dtype=bool), 350, "#be123c"),
    ]
    for name, arr, y, color in timelines:
        svg.append(f'<text x="{margin_l}" y="{y - 7}" font-family="Arial" font-size="12" fill="#555">{name}</text>')
        svg.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" y2="{y}" stroke="#ddd" />')
        for start, end in true_runs(arr):
            x1 = float(x_at(start * result.hop_s))
            x2 = float(x_at(end * result.hop_s))
            svg.append(f'<rect x="{x1:.2f}" y="{y - 9}" width="{max(1.0, x2 - x1):.2f}" height="18" fill="{color}" opacity="0.75" />')

    # Pitch curve.
    pitch = np.asarray(result.pitch_hz, dtype=np.float32)
    times = np.arange(len(pitch), dtype=np.float32) * result.hop_s
    voiced_pitch = pitch[pitch > 0]
    if len(voiced_pitch):
        fmin = max(40.0, float(np.percentile(voiced_pitch, 2)) * 0.9)
        fmax = min(2200.0, float(np.percentile(voiced_pitch, 98)) * 1.1)
        fmax = max(fmax, fmin + 20.0)
        usable = pitch > 0
        y = scale(pitch[usable], fmin, fmax, 425, 210)
        pts = [(float(x_at(t)), float(yy)) for t, yy in zip(times[usable], y)]
        svg.append(polyline(pts, "#7c3aed", 1.5))
        for hz in nice_pitch_ticks(fmin, fmax):
            yy = float(scale(np.array([hz], dtype=np.float32), fmin, fmax, 425, 210)[0])
            svg.append(f'<line x1="{margin_l}" y1="{yy:.2f}" x2="{margin_l + plot_w}" y2="{yy:.2f}" stroke="#e5e5e5" />')
            svg.append(f'<text x="24" y="{yy + 4:.2f}" font-family="Arial" font-size="11" fill="#666">{hz:.0f} Hz</text>')
    else:
        svg.append(f'<text x="{margin_l + 20}" y="535" font-family="Arial" font-size="14" fill="#777">No non-zero pitch frames available.</text>')

    # Time ticks.
    for t in np.linspace(0, duration, num=7):
        x = float(x_at(t))
        svg.append(f'<line x1="{x:.2f}" y1="645" x2="{x:.2f}" y2="652" stroke="#666" />')
        svg.append(f'<text x="{x - 14:.2f}" y="670" font-family="Arial" font-size="11" fill="#666">{t:.1f}s</text>')

    svg.append(
        '<text x="70" y="690" font-family="Arial" font-size="12" fill="#777">'
        "Confidence curve unavailable: existing inference result exposes thresholded booleans, not raw probabilities."
        "</text>"
    )
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def true_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(arr):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(arr)))
    return runs


def nice_pitch_ticks(fmin: float, fmax: float) -> list[float]:
    candidates = [55, 82, 110, 147, 196, 262, 330, 392, 523, 659, 784, 1047, 1568, 2093]
    return [float(x) for x in candidates if fmin <= x <= fmax]


def write_markdown_report(
    out_path: Path,
    audio_path: Path,
    result: Any,
    summary: dict[str, Any],
    artifacts: dict[str, str],
    conversion: dict[str, Any],
    error: str | None,
) -> None:
    lines = [
        f"# Evaluation: {audio_path.stem}",
        "",
        f"- Input: `{audio_path}`",
        f"- Audio used for inference: `{conversion.get('used_audio_path')}`",
        f"- Converted: `{conversion.get('converted')}`",
        f"- Score: `{summary.get('score')}`",
        f"- Summary: {result.summary if error is None else 'Inference failed'}",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{artifacts['json']}`",
        f"- Plot: `{artifacts.get('plot', 'not generated')}`",
        "",
        "## Key Metrics",
        "",
    ]
    metric_keys = [
        "duration_s",
        "audio_rms",
        "voiced_frame_ratio",
        "voiced_duration_s",
        "mean_f0_hz",
        "median_f0_hz",
        "min_f0_hz",
        "max_f0_hz",
        "pitch_accuracy",
        "pitch_drift_cents",
        "breath_count",
        "onset_count",
        "onset_clarity",
        "technique",
        "technique_confidence",
        "note_count",
        "voice_quality_available",
        "confidence_curve_available",
    ]
    for key in metric_keys:
        lines.append(f"- `{key}`: `{summary.get(key)}`")

    lines += [
        "",
        "## Issues",
        "",
    ]
    issues = getattr(result, "issues", []) if error is None else [error]
    if issues:
        lines += [f"- {issue}" for issue in issues]
    else:
        lines.append("- None reported.")

    lines += [
        "",
        "## Exercises",
        "",
    ]
    exercises = getattr(result, "exercises", []) if error is None else []
    if exercises:
        lines += [f"- {exercise}" for exercise in exercises]
    else:
        lines.append("- None reported.")

    lines += [
        "",
        "## Confidence Curve Availability",
        "",
        summary.get(
            "confidence_curve_note",
            "The current inference result does not expose per-frame confidence.",
        ),
        "",
    ]
    if conversion.get("conversion_error"):
        lines += [
            "## Conversion Error",
            "",
            str(conversion["conversion_error"]),
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_audio(
    audio_path: Path,
    output_dir: Path,
    checkpoint: Path | None,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_slug = audio_path.stem
    sample_dir = output_dir / sample_slug
    sample_dir.mkdir(parents=True, exist_ok=True)

    used_audio, conversion = ensure_decodable_audio(audio_path, output_dir)
    json_path = sample_dir / f"{sample_slug}.json"
    md_path = sample_dir / f"{sample_slug}.md"
    plot_path = sample_dir / f"{sample_slug}_plots.svg"

    artifacts: dict[str, str | None] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "plot": str(plot_path),
    }
    checkpoint_exists = checkpoint is not None and checkpoint.exists()
    debug = {
        "inference_mode": "checkpoint" if checkpoint_exists else "fallback",
        "checkpoint_path_used": str(checkpoint) if checkpoint_exists else None,
        "device_used": device,
        "model_stack_used": "ml_new",
    }

    payload: dict[str, Any] = {
        "sample": sample_slug,
        "input_path": str(audio_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "device": device,
        "inference_entrypoint": "ml_new.inference.coach_inference.analyse_recording",
        "debug": debug,
        "conversion": conversion,
        "artifacts": artifacts,
        "status": "error",
    }

    if conversion.get("conversion_error"):
        artifacts["plot"] = None
        payload["error"] = conversion["conversion_error"]
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        dummy = type("Result", (), {"summary": "", "issues": [], "exercises": []})()
        write_markdown_report(md_path, audio_path, dummy, {}, artifacts, conversion, payload["error"])
        return payload

    try:
        result = analyse_recording(str(used_audio), checkpoint=checkpoint, device=device)
        audio, sr, audio_error = load_waveform(used_audio)
        summary = summarize_result(result, audio, sr)
        if audio_error:
            summary["waveform_load_error"] = audio_error
        make_plot_svg(audio, sr, result, sample_slug, plot_path)
        payload.update(
            {
                "status": "success",
                "summary_metrics": summary,
                "result": clean_json(result),
                "model_output_limitations": [
                    "No raw per-frame pitch confidence is exposed by CoachingResult.",
                    "No raw per-frame VAD probability is exposed by CoachingResult.",
                    "Breath/onset arrays are thresholded booleans, not calibrated probabilities.",
                ],
            }
        )
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_markdown_report(md_path, audio_path, result, summary, artifacts, conversion, None)
        return payload
    except Exception as exc:  # pragma: no cover - reported in artifacts
        artifacts["plot"] = None
        payload["error"] = f"{type(exc).__name__}: {exc}"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        dummy = type("Result", (), {"summary": "", "issues": [], "exercises": []})()
        write_markdown_report(md_path, audio_path, dummy, {}, artifacts, conversion, payload["error"])
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio file to evaluate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eval/self_recorded"),
        help="Directory where artifacts will be written",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml_new/checkpoints/unified/best.pt"),
        help="UnifiedVocalModel checkpoint. Use an empty string for fallback mode.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, mps, cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint
    if str(checkpoint) == "":
        checkpoint = None
    elif checkpoint is not None and not checkpoint.exists():
        print(f"warning: checkpoint does not exist, inference will use fallback: {checkpoint}", file=sys.stderr)
    result = evaluate_audio(args.audio, args.output_dir, checkpoint, args.device)
    print(json.dumps({"status": result["status"], "sample": result["sample"], "artifacts": result["artifacts"]}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
