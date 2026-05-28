#!/usr/bin/env python3
"""Evaluate all self-recorded diagnostic samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_audio import evaluate_audio  # noqa: E402

EXPECTED = {
    "00_silence": "Fan/background noise only. Expected: mostly unvoiced, no stable f0, low/no singing score.",
    "01_speaking_voice": "Speech, not singing. Expected: voice detected, but singing-specific pitch/rhythm coaching should be treated cautiously.",
    "03_sustained_aaa": "Held sung vowel. Expected: sustained voiced region with stable f0 and few onsets.",
    "04_pitch_slide": "Sung pitch slide. Expected: voiced region with moving f0/pitch curve.",
    "05_twinkle_twinkle": "Short sung melody. Expected: voiced singing with multiple notes/onsets and changing f0.",
}

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".webm", ".aac", ".mp4"}


def default_samples_dir() -> Path:
    preferred = Path("samples/self_recorded")
    if preferred.exists():
        return preferred
    return Path("samples")


def find_audio_files(samples_dir: Path, extensions: set[str] | None = None) -> list[Path]:
    allowed = extensions or AUDIO_EXTS
    files = [p for p in samples_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    return sorted(files)


def expected_behavior_check(sample: str, metrics: dict) -> str:
    """Return a lightweight expectation note from observable metrics."""
    voiced_ratio = float(metrics.get("voiced_frame_ratio") or 0.0)
    onset_count = int(metrics.get("onset_count") or 0)
    note_count = int(metrics.get("note_count") or 0)
    f0_range = None
    if metrics.get("min_f0_hz") is not None and metrics.get("max_f0_hz") is not None:
        f0_range = float(metrics["max_f0_hz"]) - float(metrics["min_f0_hz"])

    if sample.startswith("00_silence"):
        return "PASS-ish" if voiced_ratio < 0.10 else "QUESTIONABLE: silence produced substantial voiced frames"
    if sample.startswith("01_speaking_voice"):
        return "OBSERVE: speech may be voiced; singing-specific outputs are not validated"
    if sample.startswith("03_sustained_aaa"):
        return "PASS-ish" if voiced_ratio > 0.50 and onset_count <= 3 else "QUESTIONABLE: sustained vowel was not cleanly represented"
    if sample.startswith("04_pitch_slide"):
        return "PASS-ish" if voiced_ratio > 0.40 and (f0_range or 0.0) > 80 else "QUESTIONABLE: pitch slide did not show clear f0 movement"
    if sample.startswith("05_twinkle_twinkle"):
        return "PASS-ish" if voiced_ratio > 0.30 and (onset_count >= 2 or note_count >= 2) else "QUESTIONABLE: melody did not show multiple events"
    return "OBSERVE"


def write_batch_summary(output_dir: Path, results: list[dict]) -> None:
    summary_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    compact = []
    for item in results:
        metrics = item.get("summary_metrics") or {}
        compact.append(
            {
                "sample": item.get("sample"),
                "status": item.get("status"),
                "expected": EXPECTED.get(item.get("sample", ""), ""),
                "expectation_check": item.get("expectation_check"),
                "score": metrics.get("score"),
                "voiced_frame_ratio": metrics.get("voiced_frame_ratio"),
                "mean_f0_hz": metrics.get("mean_f0_hz"),
                "min_f0_hz": metrics.get("min_f0_hz"),
                "max_f0_hz": metrics.get("max_f0_hz"),
                "breath_count": metrics.get("breath_count"),
                "onset_count": metrics.get("onset_count"),
                "note_count": metrics.get("note_count"),
                "error": item.get("error"),
                "artifacts": item.get("artifacts"),
            }
        )
    summary_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")

    lines = [
        "# Self-Recorded Evaluation Summary",
        "",
        f"- Samples evaluated: `{len(results)}`",
        f"- JSON summary: `{summary_path}`",
        "",
        "| Sample | Status | Expected behavior check | Score | Voiced ratio | Mean f0 | Onsets | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in compact:
        mean_f0 = item["mean_f0_hz"]
        mean_f0_s = "" if mean_f0 is None else f"{mean_f0:.1f}"
        lines.append(
            "| {sample} | {status} | {check} | {score} | {vr:.3f} | {f0} | {onsets} | {notes} |".format(
                sample=item["sample"],
                status=item["status"],
                check=(item["expectation_check"] or "").replace("|", "\\|"),
                score=item["score"],
                vr=float(item["voiced_frame_ratio"] or 0.0),
                f0=mean_f0_s,
                onsets=item["onset_count"],
                notes=item["note_count"],
            )
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The existing inference entrypoint exposes thresholded voiced/breath/onset arrays, not raw confidence curves.",
        "- `.m4a` files are converted with macOS `afconvert` when direct decoding is unavailable.",
        "- Checks are heuristics for diagnostic sanity, not formal model accuracy metrics.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=default_samples_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eval/self_recorded"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Optional file extensions to include, e.g. --extensions .wav",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extensions = None
    if args.extensions is not None:
        extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in args.extensions
        }
    files = find_audio_files(args.samples_dir, extensions)
    if not files:
        print(f"No audio files found under {args.samples_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in files:
        print(f"Evaluating {path}")
        result = evaluate_audio(path, args.output_dir, args.checkpoint, args.device)
        result["expected_behavior"] = EXPECTED.get(result.get("sample", ""), "")
        if result.get("status") == "success":
            result["expectation_check"] = expected_behavior_check(
                result.get("sample", ""),
                result.get("summary_metrics") or {},
            )
        else:
            result["expectation_check"] = "NOT EVALUATED: inference did not run"
        results.append(result)
    write_batch_summary(args.output_dir, results)
    print(json.dumps({"status": "complete", "count": len(results), "output_dir": str(args.output_dir)}, indent=2))
    return 0 if all(r.get("status") == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
