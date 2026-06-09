#!/usr/bin/env python3
"""Generate synthetic diagnostic WAVs and compare checkpoint vs pyin baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import HOP_S, SR  # noqa: E402
from scripts.eval.audit_model_outputs import run_raw_checkpoint  # noqa: E402
from scripts.eval.compare_baseline_outputs import f0_metrics, run_pyin_baseline  # noqa: E402


SPECS = [
    {
        "name": "digital_silence_5s",
        "kind": "silence",
        "expected_f0_hz": None,
        "should_be_voiced": False,
    },
    {
        "name": "white_noise_5s",
        "kind": "white_noise",
        "expected_f0_hz": None,
        "should_be_voiced": False,
    },
    {
        "name": "low_hum_80hz_5s",
        "kind": "sine",
        "freq_hz": 80.0,
        "expected_f0_hz": 80.0,
        "should_be_voiced": True,
    },
    {
        "name": "sine_220hz_5s",
        "kind": "sine",
        "freq_hz": 220.0,
        "expected_f0_hz": 220.0,
        "should_be_voiced": True,
    },
    {
        "name": "sine_440hz_5s",
        "kind": "sine",
        "freq_hz": 440.0,
        "expected_f0_hz": 440.0,
        "should_be_voiced": True,
    },
    {
        "name": "sine_sweep_220_to_440_5s",
        "kind": "sweep",
        "start_hz": 220.0,
        "end_hz": 440.0,
        "expected_f0_hz": 330.0,
        "should_be_voiced": True,
    },
    {
        "name": "pulsed_220hz_voiced_unvoiced",
        "kind": "pulsed",
        "freq_hz": 220.0,
        "expected_f0_hz": 220.0,
        "should_be_voiced": True,
    },
]


def generate_audio(spec: dict[str, Any], duration_s: float = 5.0) -> np.ndarray:
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    kind = spec["kind"]
    if kind == "silence":
        audio = np.zeros(n, dtype=np.float64)
    elif kind == "white_noise":
        rng = np.random.default_rng(12345)
        audio = rng.normal(0.0, 0.08, size=n)
    elif kind == "sine":
        audio = 0.25 * np.sin(2.0 * np.pi * float(spec["freq_hz"]) * t)
    elif kind == "sweep":
        start = float(spec["start_hz"])
        end = float(spec["end_hz"])
        duration = max(duration_s, 1e-6)
        phase = 2.0 * np.pi * (start * t + 0.5 * (end - start) / duration * t * t)
        audio = 0.25 * np.sin(phase)
    elif kind == "pulsed":
        carrier = 0.25 * np.sin(2.0 * np.pi * float(spec["freq_hz"]) * t)
        cycle_s = 1.0
        duty = 0.5
        gate = ((t % cycle_s) < (cycle_s * duty)).astype(np.float64)
        audio = carrier * gate
    else:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    fade_len = min(int(0.02 * SR), len(audio) // 20)
    if fade_len > 0 and kind not in {"silence", "white_noise", "pulsed"}:
        fade = np.linspace(0.0, 1.0, fade_len)
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]
    return np.clip(audio, -0.99, 0.99).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())


def cents_error(predicted_hz: float | None, expected_hz: float | None) -> float | None:
    if predicted_hz is None or expected_hz is None or predicted_hz <= 0 or expected_hz <= 0:
        return None
    return float(1200.0 * math.log2(predicted_hz / expected_hz))


def summarize_checkpoint(audio: np.ndarray, checkpoint: Path, device: str, expected_f0: float | None) -> dict[str, Any]:
    raw = run_raw_checkpoint(audio, checkpoint, device)
    metrics = f0_metrics(
        raw["raw_f0_voiced_thresholded_hz"],
        raw["voiced_prob"],
        raw["pitch_confidence"],
    )
    metrics["predicted_median_f0_hz"] = metrics["median_f0_hz"]
    metrics["f0_error_cents"] = cents_error(metrics["median_f0_hz"], expected_f0)
    metrics["confidence"] = metrics["pitch_confidence_mean"]
    metrics["false_voiced_percentage"] = metrics["voiced_frame_percentage"]
    return metrics


def summarize_baseline(audio: np.ndarray, expected_f0: float | None) -> dict[str, Any]:
    raw = run_pyin_baseline(audio)
    metrics = f0_metrics(raw["f0_hz"], raw["voiced_prob"], None)
    metrics["predicted_median_f0_hz"] = metrics["median_f0_hz"]
    metrics["f0_error_cents"] = cents_error(metrics["median_f0_hz"], expected_f0)
    metrics["confidence"] = metrics["median_voiced_probability"]
    metrics["false_voiced_percentage"] = metrics["voiced_frame_percentage"]
    return metrics


def choose_more_reliable(spec: dict[str, Any], checkpoint: dict[str, Any], baseline: dict[str, Any]) -> str:
    expected = spec["expected_f0_hz"]
    if not spec["should_be_voiced"]:
        ck_false = checkpoint["false_voiced_percentage"]
        py_false = baseline["false_voiced_percentage"]
        if py_false == 0.0 and ck_false > 0.0:
            return "baseline"
        if ck_false == 0.0 and py_false > 0.0:
            return "checkpoint"
        if abs(ck_false - py_false) < 0.05:
            return "neither"
        return "baseline" if py_false < ck_false else "checkpoint"

    def score(m: dict[str, Any]) -> float:
        voiced = float(m["voiced_frame_percentage"])
        error = abs(float(m["f0_error_cents"])) if m["f0_error_cents"] is not None else 1200.0
        jumps = float(m["semitone_jump_count"])
        coverage_penalty = abs(0.95 - voiced) * 400.0
        return error + jumps * 30.0 + coverage_penalty

    ck_score = score(checkpoint)
    py_score = score(baseline)
    if abs(ck_score - py_score) < 50.0:
        return "mixed"
    return "checkpoint" if ck_score < py_score else "baseline"


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "null"
    return f"{100.0 * float(value):.1f}%"


def write_report(path: Path, rows: list[dict[str, Any]], samples_dir: Path, output_dir: Path) -> None:
    lines = [
        "# M2 Synthetic Model Tests",
        "",
        "Goal: test checkpoint raw VAD/f0 behavior against a DSP baseline on controlled synthetic audio.",
        "",
        "No retraining, model architecture changes, scoring tuning, or P4 regression expectation changes were made.",
        "",
        "## Generated WAV Files",
        "",
        f"- Directory: `{samples_dir}`",
    ]
    for item in rows:
        lines.append(f"- `{item['wav_path']}`")
    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON summary: `{output_dir / 'summary.json'}`",
        "",
        "## Results",
        "",
        "| Sample | Method | Expected f0 | Predicted median f0 | Error cents | Voiced % | False voiced % | Jumps | Confidence | More reliable |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in rows:
        expected = item["expected_f0_hz"]
        for method_key, label in (("checkpoint", "checkpoint"), ("baseline", "baseline pyin")):
            m = item[method_key]
            false_voiced = m["false_voiced_percentage"] if not item["should_be_voiced"] else None
            lines.append(
                "| {sample} | {method} | {expected} | {pred} | {err} | {voiced} | {false_voiced} | {jumps} | {conf} | {winner} |".format(
                    sample=f"`{item['sample']}`",
                    method=label,
                    expected=fmt(expected),
                    pred=fmt(m["predicted_median_f0_hz"]),
                    err=fmt(m["f0_error_cents"]),
                    voiced=pct(m["voiced_frame_percentage"]),
                    false_voiced=pct(false_voiced) if false_voiced is not None else "",
                    jumps=m["semitone_jump_count"],
                    conf=fmt(m["confidence"]),
                    winner=item["more_reliable"] if method_key == "checkpoint" else "",
                )
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `digital_silence_5s` and `white_noise_5s` should have near-zero voiced output. Any voiced percentage there is false voiced behavior.",
        "- Pure sine waves are not human singing, but they are useful for measuring f0 binning, octave errors, and confidence.",
        "- `sine_sweep_220_to_440_5s` uses an expected median f0 of 330 Hz for the cents-error column; direction and continuity matter more than that single number.",
        "- `pulsed_220hz_voiced_unvoiced.wav` should show roughly half voiced coverage if the detector respects the silent gaps.",
        "",
        "## Overall Takeaway",
        "",
        overall_takeaway(rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def overall_takeaway(rows: list[dict[str, Any]]) -> str:
    winners = [row["more_reliable"] for row in rows]
    baseline_wins = winners.count("baseline")
    checkpoint_wins = winners.count("checkpoint")
    mixed = winners.count("mixed") + winners.count("neither")
    return (
        f"Baseline is more reliable on `{baseline_wins}` synthetic cases, checkpoint on `{checkpoint_wins}`, with `{mixed}` mixed/neither cases. "
        "These tests reinforce the M1 recommendation: use a hybrid/ensemble for now, with DSP checks guarding checkpoint VAD/f0 before product-facing scoring or coaching."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=Path("samples/synthetic_model_tests"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/synthetic_model_tests"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in SPECS:
        name = spec["name"]
        audio = generate_audio(spec)
        wav_path = args.samples_dir / f"{name}.wav"
        write_wav(wav_path, audio)
        print(f"Testing {wav_path}")
        checkpoint_metrics = summarize_checkpoint(audio, args.checkpoint, args.device, spec["expected_f0_hz"])
        baseline_metrics = summarize_baseline(audio, spec["expected_f0_hz"])
        row = {
            "sample": name,
            "wav_path": str(wav_path),
            "expected_f0_hz": spec["expected_f0_hz"],
            "should_be_voiced": spec["should_be_voiced"],
            "checkpoint": checkpoint_metrics,
            "baseline": baseline_metrics,
        }
        row["more_reliable"] = choose_more_reliable(spec, checkpoint_metrics, baseline_metrics)
        rows.append(row)

    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(Path("M2_SYNTHETIC_MODEL_TESTS.md"), rows, args.samples_dir, args.output_dir)
    print(json.dumps({"status": "complete", "samples": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
