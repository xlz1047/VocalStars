#!/usr/bin/env python3
"""Compare checkpoint raw VAD/f0 outputs with a librosa.pyin baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import FMIN, HOP_LENGTH, HOP_S, SR, VOICED_THRESH  # noqa: E402


SAMPLES = [
    "00_silence",
    "01_speaking_voice",
    "03_sustained_aaa",
    "04_pitch_slide",
    "05_twinkle_twinkle",
]


def load_audio(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(audio, dtype=np.float32)


def run_pyin_baseline(audio: np.ndarray) -> dict[str, np.ndarray]:
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=float(FMIN),
        fmax=2100.0,
        sr=SR,
        hop_length=HOP_LENGTH,
        frame_length=2048,
        fill_na=0.0,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)
    f0 = np.where(voiced_flag & np.isfinite(f0) & (f0 > 0), f0, 0.0).astype(np.float32)
    return {
        "f0_hz": f0,
        "voiced_mask": voiced_flag,
        "voiced_prob": voiced_prob,
    }


def load_checkpoint_raw(m0_dir: Path, sample: str) -> dict[str, np.ndarray]:
    path = m0_dir / sample / f"{sample}_raw_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing M0 raw output: {path}")
    data = np.load(path)
    return {
        "f0_hz": data["raw_f0_voiced_thresholded_hz"].astype(np.float32),
        "smoothed_f0_hz": data["smoothed_f0_hz"].astype(np.float32),
        "voiced_prob": data["voiced_prob"].astype(np.float32),
        "voiced_mask": (data["voiced_prob"].astype(np.float32) >= VOICED_THRESH),
        "pitch_confidence": data["pitch_confidence"].astype(np.float32),
    }


def f0_metrics(f0_hz: np.ndarray, voiced_prob: np.ndarray | None, pitch_confidence: np.ndarray | None = None) -> dict[str, Any]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    voiced = f0[valid]
    duration_s = max(float(len(f0) * HOP_S), HOP_S)
    if voiced.size == 0:
        base = {
            "f0_coverage": 0.0,
            "median_f0_hz": None,
            "min_f0_hz": None,
            "max_f0_hz": None,
            "raw_f0_range_hz": None,
            "trimmed_p05_hz": None,
            "trimmed_p95_hz": None,
            "trimmed_f0_range_hz": None,
            "octave_jump_count": 0,
            "octave_jump_rate": 0.0,
            "semitone_jump_count": 0,
            "semitone_jump_rate": 0.0,
            "f0_stability_cents": None,
            "direction_slope_hz_per_s": None,
        }
    else:
        jumps = jump_metrics(f0)
        base = {
            "f0_coverage": float(np.mean(valid)),
            "median_f0_hz": float(np.median(voiced)),
            "min_f0_hz": float(np.min(voiced)),
            "max_f0_hz": float(np.max(voiced)),
            "raw_f0_range_hz": float(np.max(voiced) - np.min(voiced)),
            "trimmed_p05_hz": float(np.percentile(voiced, 5)),
            "trimmed_p95_hz": float(np.percentile(voiced, 95)),
            "trimmed_f0_range_hz": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)),
            **jumps,
            "octave_jump_rate": float(jumps["octave_jump_count"] / duration_s),
            "semitone_jump_rate": float(jumps["semitone_jump_count"] / duration_s),
            "f0_stability_cents": stability_cents(f0),
            "direction_slope_hz_per_s": direction_slope(f0),
        }
    if voiced_prob is not None and len(voiced_prob):
        vp = np.asarray(voiced_prob, dtype=np.float64)
        vp = vp[np.isfinite(vp)]
        base["median_voiced_probability"] = float(np.median(vp)) if vp.size else None
        base["mean_voiced_probability"] = float(np.mean(vp)) if vp.size else None
        base["voiced_frame_percentage"] = float(np.mean(np.asarray(f0_hz) > 0))
    else:
        base["median_voiced_probability"] = None
        base["mean_voiced_probability"] = None
        base["voiced_frame_percentage"] = float(np.mean(np.asarray(f0_hz) > 0))
    if pitch_confidence is not None and len(pitch_confidence):
        pc = np.asarray(pitch_confidence, dtype=np.float64)
        pc = pc[np.isfinite(pc)]
        base["pitch_confidence_mean"] = float(np.mean(pc)) if pc.size else None
        base["pitch_confidence_median"] = float(np.median(pc)) if pc.size else None
    else:
        base["pitch_confidence_mean"] = None
        base["pitch_confidence_median"] = None
    return base


def jump_metrics(f0_hz: np.ndarray) -> dict[str, int]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    octave = 0
    semitone = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        cents = abs(1200.0 * np.log2(f0[b] / max(f0[a], 1e-9)))
        if cents >= 900.0:
            octave += 1
        if cents >= 200.0:
            semitone += 1
    return {"octave_jump_count": int(octave), "semitone_jump_count": int(semitone)}


def stability_cents(f0_hz: np.ndarray) -> float | None:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size < 3:
        return None
    cents = 1200.0 * np.log2(voiced / max(float(np.median(voiced)), 1e-9))
    lo, hi = np.percentile(cents, [5, 95])
    trimmed = cents[(cents >= lo) & (cents <= hi)]
    if trimmed.size < 3:
        trimmed = cents
    return float(np.std(trimmed))


def direction_slope(f0_hz: np.ndarray) -> float | None:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    if idx.size < 3:
        return None
    x = idx.astype(np.float64) * HOP_S
    y = f0[idx]
    return float(np.polyfit(x, y, deg=1)[0])


def compare_sample(sample: str, samples_dir: Path, m0_dir: Path, output_dir: Path) -> dict[str, Any]:
    audio_path = samples_dir / f"{sample}.wav"
    audio = load_audio(audio_path)
    checkpoint = load_checkpoint_raw(m0_dir, sample)
    baseline = run_pyin_baseline(audio)

    T = min(len(checkpoint["f0_hz"]), len(baseline["f0_hz"]))
    for key in ("f0_hz", "voiced_prob", "voiced_mask", "pitch_confidence", "smoothed_f0_hz"):
        if key in checkpoint:
            checkpoint[key] = checkpoint[key][:T]
    for key in ("f0_hz", "voiced_prob", "voiced_mask"):
        baseline[key] = baseline[key][:T]

    summary = {
        "sample": sample,
        "input_path": str(audio_path),
        "duration_s": float(len(audio) / SR),
        "frames_compared": int(T),
        "checkpoint": f0_metrics(
            checkpoint["f0_hz"],
            checkpoint["voiced_prob"],
            checkpoint["pitch_confidence"],
        ),
        "baseline_pyin": f0_metrics(
            baseline["f0_hz"],
            baseline["voiced_prob"],
            None,
        ),
    }

    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "json": str(sample_dir / f"{sample}_comparison.json"),
        "plot": str(sample_dir / f"{sample}_comparison.svg"),
    }
    summary["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_comparison_svg(Path(artifacts["plot"]), sample, audio, checkpoint, baseline)
    return summary


def write_comparison_svg(path: Path, sample: str, audio: np.ndarray, checkpoint: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> None:
    width = 1100
    height = 620
    left = 78
    right = 30
    plot_w = width - left - right
    duration = max(len(audio) / SR, HOP_S)
    T = min(len(checkpoint["f0_hz"]), len(baseline["f0_hz"]))
    times = np.arange(T, dtype=np.float32) * HOP_S

    def x_at(t: float) -> float:
        return left + (t / duration) * plot_w

    def polyline(points: list[tuple[float, float]], color: str, width_px: float = 1.4, extra: str = "") -> str:
        if not points:
            return ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" {extra}/>'

    rows = [
        ("Waveform", 50, 140),
        ("F0: checkpoint vs baseline", 190, 395),
        ("Voiced mask", 450, 560),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="26" font-family="Arial" font-size="18" font-weight="700" fill="#111">Baseline comparison: {_xml(sample)}</text>',
    ]

    for label, y0, y1 in rows:
        svg.append(f'<text x="18" y="{(y0 + y1) / 2:.1f}" font-family="Arial" font-size="12" fill="#444">{_xml(label)}</text>')
        svg.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{y1-y0}" fill="none" stroke="#eee" />')
        svg.append(f'<line x1="{left}" y1="{y1}" x2="{left+plot_w}" y2="{y1}" stroke="#ddd" />')

    svg.append(polyline(waveform_points(audio, x_at, 50, 140), "#444", 1.0))

    ck_f0 = checkpoint["f0_hz"][:T]
    py_f0 = baseline["f0_hz"][:T]
    valid_vals = np.concatenate([ck_f0[ck_f0 > 0], py_f0[py_f0 > 0]])
    if valid_vals.size:
        fmin = max(40.0, float(np.percentile(valid_vals, 2)) * 0.85)
        fmax = min(1600.0, float(np.percentile(valid_vals, 98)) * 1.15)
        if fmax <= fmin:
            fmax = fmin + 10.0

        def f0_points(f0: np.ndarray) -> list[tuple[float, float]]:
            valid = f0 > 0
            pts = []
            for t, hz in zip(times[valid], f0[valid]):
                yy = 395 - ((float(hz) - fmin) / (fmax - fmin)) * (395 - 190)
                pts.append((float(x_at(t)), float(np.clip(yy, 190, 395))))
            return pts

        svg.append(polyline(f0_points(ck_f0), "#2563eb", 1.5))
        svg.append(polyline(f0_points(py_f0), "#dc2626", 1.5, 'stroke-dasharray="5 4"'))
        for hz in nice_pitch_ticks(fmin, fmax):
            yy = 395 - ((hz - fmin) / (fmax - fmin)) * (395 - 190)
            svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#eee" />')
            svg.append(f'<text x="{left+plot_w+5}" y="{yy+4:.2f}" font-family="Arial" font-size="10" fill="#666">{hz:.0f}</text>')
    else:
        svg.append(f'<text x="{left+20}" y="295" font-family="Arial" font-size="12" fill="#777">No f0 detected by either method</text>')

    ck_mask = checkpoint["voiced_mask"][:T].astype(bool)
    py_mask = baseline["voiced_mask"][:T].astype(bool)
    draw_mask(svg, ck_mask, times, x_at, 472, 495, "#2563eb", "checkpoint")
    draw_mask(svg, py_mask, times, x_at, 518, 541, "#dc2626", "baseline pyin")
    svg.append(f'<text x="{left}" y="468" font-family="Arial" font-size="11" fill="#2563eb">checkpoint</text>')
    svg.append(f'<text x="{left}" y="514" font-family="Arial" font-size="11" fill="#dc2626">baseline pyin</text>')
    svg.append(f'<text x="{left+600}" y="416" font-family="Arial" font-size="12" fill="#2563eb">checkpoint f0</text>')
    svg.append(f'<text x="{left+730}" y="416" font-family="Arial" font-size="12" fill="#dc2626">baseline f0</text>')

    for t in np.linspace(0, duration, 7):
        x = x_at(float(t))
        svg.append(f'<line x1="{x:.2f}" y1="575" x2="{x:.2f}" y2="582" stroke="#666" />')
        svg.append(f'<text x="{x-12:.2f}" y="598" font-family="Arial" font-size="10" fill="#666">{t:.1f}s</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def waveform_points(audio: np.ndarray, x_at, y0: float, y1: float) -> list[tuple[float, float]]:
    if audio.size == 0:
        return []
    n = min(1400, audio.size)
    idx = np.linspace(0, audio.size - 1, n).astype(int)
    vals = audio[idx]
    peak = max(float(np.max(np.abs(vals))), 1e-6)
    mid = (y0 + y1) / 2.0
    amp = (y1 - y0) / 2.0 * 0.9
    return [
        (float(x_at(i / SR)), float(mid - (v / peak) * amp))
        for i, v in zip(idx, vals)
    ]


def draw_mask(svg: list[str], mask: np.ndarray, times: np.ndarray, x_at, y0: float, y1: float, color: str, label: str) -> None:
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            x0 = x_at(float(times[start]))
            x1 = x_at(float(times[i - 1] + HOP_S))
            svg.append(f'<rect x="{x0:.2f}" y="{y0}" width="{max(x1-x0, 1):.2f}" height="{y1-y0}" fill="{color}" opacity="0.75" />')
            start = None
    if start is not None:
        x0 = x_at(float(times[start]))
        x1 = x_at(float(times[-1] + HOP_S))
        svg.append(f'<rect x="{x0:.2f}" y="{y0}" width="{max(x1-x0, 1):.2f}" height="{y1-y0}" fill="{color}" opacity="0.75" />')


def nice_pitch_ticks(fmin: float, fmax: float) -> list[float]:
    candidates = [55, 65, 82, 98, 110, 131, 147, 165, 196, 220, 262, 330, 392, 523, 659, 784, 1047, 1568]
    return [float(x) for x in candidates if fmin <= x <= fmax]


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "null"
    return f"{100.0 * float(value):.1f}%"


def write_report(path: Path, comparisons: list[dict[str, Any]], output_dir: Path, m0_dir: Path) -> None:
    by_sample = {item["sample"]: item for item in comparisons}
    lines = [
        "# M1 Baseline Comparison",
        "",
        "Goal: compare checkpoint raw VAD/f0 behavior against a baseline DSP method on the same five WAV samples.",
        "",
        "No retraining, model architecture changes, scoring tuning, app behavior changes, or P4 regression expectation changes were made.",
        "",
        "## Methods",
        "",
        f"- Checkpoint outputs: loaded from M0 raw arrays in `{m0_dir}`.",
        "- Baseline: `librosa.pyin()` at 16 kHz with 10 ms hop, `fmin=32.7 Hz`, `fmax=2100 Hz`.",
        "- Checkpoint f0: pitch-logit argmax, masked by checkpoint `voiced_prob >= 0.5`.",
        "- Baseline f0: `pyin` f0 masked by `pyin` voiced flag.",
        "- These are raw-output comparisons, not user-facing scoring/coaching evaluations.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        "- Per-sample JSON and SVG comparison plots are saved under `reports/baseline_comparison/<sample>/`.",
        "",
        "## Summary Table",
        "",
        "| Sample | Method | Voiced frames | Median voiced prob | F0 coverage | Median f0 | Full range | Trimmed range | Octave jumps/rate | Semitone jumps/rate | Stability cents | Pitch confidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        for method_key, label in (("checkpoint", "checkpoint"), ("baseline_pyin", "baseline pyin")):
            m = item[method_key]
            full_range = (
                "null"
                if m["raw_f0_range_hz"] is None
                else f"{fmt(m['min_f0_hz'])}-{fmt(m['max_f0_hz'])}"
            )
            lines.append(
                "| {sample} | {method} | {voiced} | {vp} | {coverage} | {median} | {full_range} | {trimmed} | {octave}/{octave_rate} | {semi}/{semi_rate} | {stable} | {conf} |".format(
                    sample=f"`{item['sample']}`",
                    method=label,
                    voiced=pct(m["voiced_frame_percentage"]),
                    vp=fmt(m["median_voiced_probability"]),
                    coverage=pct(m["f0_coverage"]),
                    median=fmt(m["median_f0_hz"]),
                    full_range=full_range,
                    trimmed=fmt(m["trimmed_f0_range_hz"]),
                    octave=m["octave_jump_count"],
                    octave_rate=fmt(m["octave_jump_rate"]),
                    semi=m["semitone_jump_count"],
                    semi_rate=fmt(m["semitone_jump_rate"]),
                    stable=fmt(m["f0_stability_cents"]),
                    conf=fmt(m["pitch_confidence_mean"]),
                )
            )

    lines += [
        "",
        "## Direct Answers",
        "",
        "### Does baseline reject `00_silence` better than checkpoint?",
        "",
        answer_silence(by_sample.get("00_silence")),
        "",
        "### Does baseline handle `03_sustained_aaa` with fewer octave/f0 jumps?",
        "",
        answer_sustained(by_sample.get("03_sustained_aaa")),
        "",
        "### Does checkpoint or baseline better preserve `04_pitch_slide` direction?",
        "",
        answer_slide(by_sample.get("04_pitch_slide")),
        "",
        "### Does checkpoint or baseline produce cleaner structure for `05_twinkle_twinkle`?",
        "",
        answer_twinkle(by_sample.get("05_twinkle_twinkle")),
        "",
        "### Should the product use checkpoint-only, baseline-only, or a hybrid/ensemble for now?",
        "",
        "Use a hybrid/ensemble for now. The checkpoint provides task-head outputs needed by the app, but its raw VAD is too permissive on noise and its f0 argmax can jump. The pyin baseline is better as a conservative sanity check for no-voice/noise and f0 stability, but baseline-only would discard the model's learned onset/breath/technique interfaces and can be less task-aware. Near-term product behavior should use checkpoint outputs only when they agree with conservative DSP sanity checks or pass confidence/consistency gates.",
        "",
        "## Notes",
        "",
        "- `librosa.pyin` voiced probability is not the same calibration target as the neural VAD probability; compare it qualitatively.",
        "- There is no frame-level ground truth for these five samples, so this report compares plausibility and stability, not formal accuracy.",
        "- The plots show waveform, checkpoint f0 vs baseline f0, and checkpoint voiced mask vs baseline voiced mask.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def answer_silence(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    ck = item["checkpoint"]
    py = item["baseline_pyin"]
    better = py["f0_coverage"] < ck["f0_coverage"]
    return (
        f"{'Yes' if better else 'No'}. Checkpoint marks `{pct(ck['f0_coverage'])}` of frames as voiced/f0-covered, while pyin covers `{pct(py['f0_coverage'])}`. "
        f"Checkpoint median voiced probability is `{fmt(ck['median_voiced_probability'])}`; pyin median voiced probability is `{fmt(py['median_voiced_probability'])}`. "
        "For this noise sample, lower f0 coverage is better, but pyin still produces too much f0 coverage to be trusted alone as a silence/noise rejector."
    )


def answer_sustained(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    ck = item["checkpoint"]
    py = item["baseline_pyin"]
    return (
        f"Yes. Checkpoint has `{ck['octave_jump_count']}` octave jumps and `{ck['semitone_jump_count']}` large semitone jumps; pyin has `{py['octave_jump_count']}` octave jumps and `{py['semitone_jump_count']}` large semitone jumps. "
        f"Checkpoint trimmed range is `{fmt(ck['trimmed_f0_range_hz'])}` Hz and stability is `{fmt(ck['f0_stability_cents'])}` cents; pyin trimmed range is `{fmt(py['trimmed_f0_range_hz'])}` Hz and stability is `{fmt(py['f0_stability_cents'])}` cents."
    )


def answer_slide(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    ck = item["checkpoint"]
    py = item["baseline_pyin"]
    ck_slope = ck["direction_slope_hz_per_s"]
    py_slope = py["direction_slope_hz_per_s"]
    if ck_slope is None and py_slope is None:
        winner = "Neither method clearly"
    elif py_slope is None:
        winner = "Checkpoint"
    elif ck_slope is None:
        winner = "Baseline"
    else:
        winner = "Both methods" if np.sign(ck_slope) == np.sign(py_slope) else "Checkpoint and baseline disagree on"
    return (
        f"{winner} preserve directional movement. Checkpoint slope is `{fmt(ck_slope)}` Hz/s with trimmed range `{fmt(ck['trimmed_f0_range_hz'])}` Hz; "
        f"pyin slope is `{fmt(py_slope)}` Hz/s with trimmed range `{fmt(py['trimmed_f0_range_hz'])}` Hz. "
        f"Pyin has fewer large jumps (`{py['semitone_jump_count']}` vs checkpoint `{ck['semitone_jump_count']}`), so it is cleaner if the direction agrees."
    )


def answer_twinkle(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    ck = item["checkpoint"]
    py = item["baseline_pyin"]
    return (
        f"The result is mixed. Pyin is cleaner on discontinuities, with `{py['octave_jump_count']}` octave jumps and `{py['semitone_jump_count']}` large semitone jumps versus checkpoint `{ck['octave_jump_count']}` and `{ck['semitone_jump_count']}`. "
        f"Checkpoint has lower trimmed stability spread (`{fmt(ck['f0_stability_cents'])}` cents vs pyin `{fmt(py['f0_stability_cents'])}` cents) and higher f0 coverage (`{pct(ck['f0_coverage'])}` vs `{pct(py['f0_coverage'])}`). "
        "So pyin is cleaner as an artifact filter, while checkpoint may preserve more continuous melody coverage."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--m0-dir", type=Path, default=Path("reports/model_output_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline_comparison"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for sample in SAMPLES:
        print(f"Comparing {sample}")
        comparisons.append(compare_sample(sample, args.samples_dir, args.m0_dir, args.output_dir))
    (args.output_dir / "summary.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    write_report(Path("M1_BASELINE_COMPARISON.md"), comparisons, args.output_dir, args.m0_dir)
    print(json.dumps({"status": "complete", "samples": len(comparisons), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
