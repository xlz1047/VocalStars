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

TASK_CONFIGS = {
    "00_silence": {
        "task_type": "free_singing",
        "skill_focus": "invalid_input_regression",
        "target": None,
        "reference": None,
        "scoring_mode": "auto",
        "strictness": "normal",
    },
    "01_speaking_voice": {
        "task_type": "free_singing",
        "skill_focus": "invalid_input_regression",
        "target": None,
        "reference": None,
        "scoring_mode": "auto",
        "strictness": "normal",
    },
    "03_sustained_aaa": {
        "task_type": "sustained_note",
        "skill_focus": "pitch_stability",
        "target": {"vowel": "aa"},
        "reference": None,
        "scoring_mode": "diagnostic",
        "strictness": "normal",
    },
    "04_pitch_slide": {
        "task_type": "pitch_slide",
        "skill_focus": "smooth_continuous_slide",
        "target": {"direction": "up_or_down"},
        "reference": None,
        "scoring_mode": "diagnostic",
        "strictness": "normal",
    },
    "05_twinkle_twinkle": {
        "task_type": "free_singing",
        "skill_focus": "general",
        "target": None,
        "reference": None,
        "scoring_mode": "general",
        "strictness": "normal",
    },
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
    validity = metrics.get("analysis_validity") or {}
    input_type = validity.get("input_type")
    is_analyzable = bool(validity.get("is_analyzable"))
    voiced_ratio = float(metrics.get("voiced_frame_ratio") or 0.0)
    onset_count = int(metrics.get("onset_count") or 0)
    note_count = int(metrics.get("note_count") or 0)
    f0_range = None
    if metrics.get("min_f0_hz") is not None and metrics.get("max_f0_hz") is not None:
        f0_range = float(metrics["max_f0_hz"]) - float(metrics["min_f0_hz"])

    if sample.startswith("00_silence"):
        return "PASS" if input_type != "analyzable_singing" else "FAIL: silence was treated as analyzable singing"
    if sample.startswith("01_speaking_voice"):
        return "PASS" if not is_analyzable else "FAIL: speech received normal singing coaching"
    if sample.startswith("03_sustained_aaa"):
        return (
            "PASS"
            if input_type in {"diagnostic_sustained_tone", "analyzable_singing"}
            else "QUESTIONABLE: sustained vowel was rejected as invalid"
        )
    if sample.startswith("04_pitch_slide"):
        return (
            "PASS"
            if input_type in {"diagnostic_pitch_slide", "analyzable_singing"}
            else "QUESTIONABLE: pitch slide was rejected as invalid"
        )
    if sample.startswith("05_twinkle_twinkle"):
        task_status = (metrics.get("task_analysis") or {}).get("status")
        return (
            "PASS"
            if input_type == "analyzable_singing" and task_status == "free_singing_general_feedback"
            else "QUESTIONABLE: melody was not free-singing analyzable"
        )
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
                "summary": (item.get("result") or {}).get("summary"),
                "score": metrics.get("score"),
                "full_song_score": metrics.get("full_song_score"),
                "diagnostic_score": metrics.get("diagnostic_score"),
                "score_status": metrics.get("score_status"),
                "voiced_frame_ratio": metrics.get("voiced_frame_ratio"),
                "mean_f0_hz": metrics.get("mean_f0_hz"),
                "min_f0_hz": metrics.get("min_f0_hz"),
                "max_f0_hz": metrics.get("max_f0_hz"),
                "breath_count": metrics.get("breath_count"),
                "onset_count": metrics.get("onset_count"),
                "note_count": metrics.get("note_count"),
                "diagnostics": metrics.get("diagnostics"),
                "analysis_validity": metrics.get("analysis_validity"),
                "task_config": metrics.get("task_config"),
                "task_analysis": metrics.get("task_analysis"),
                "raw_note_count": _get_nested(
                    metrics.get("diagnostics") or {},
                    "note_postprocessing.raw_note_count",
                ),
                "postprocessed_note_count": _get_nested(
                    metrics.get("diagnostics") or {},
                    "note_postprocessing.postprocessed_note_count",
                ),
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
        "| Sample | Provided task | Detected input | Score status | Full-song score | Diagnostic score | Task summary | Caveats | Regression expectation |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in compact:
        validity = item.get("analysis_validity") or {}
        task_config = item.get("task_config") or {}
        task_analysis = item.get("task_analysis") or {}
        lines.append(
            "| {sample} | {task_type} | {input_type} | {score_status} | {full_score} | {diag_score} | {task_summary} | {caveats} | {check} |".format(
                sample=item["sample"],
                task_type=task_config.get("task_type", ""),
                input_type=validity.get("input_type", ""),
                full_score=_format_optional(item.get("full_song_score")),
                diag_score=_format_optional(item.get("diagnostic_score")),
                score_status=item.get("score_status", ""),
                task_summary=(task_analysis.get("summary") or item.get("summary") or "").replace("|", "\\|"),
                caveats=", ".join(task_analysis.get("caveats") or []).replace("|", "\\|"),
                check=(item["expectation_check"] or "").replace("|", "\\|"),
            )
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Analysis validity is a postprocessing gate; raw frame outputs and notes remain present for inspection.",
        "- Each sample is evaluated with an explicit task_config.",
        "- Full-song score and diagnostic score are reported separately.",
        "- Normal singing coaching is blocked for non-analyzable and diagnostic inputs.",
        "- `.m4a` files are converted with macOS `afconvert` when direct decoding is unavailable.",
        "- Checks are heuristics for diagnostic sanity, not formal model accuracy metrics.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _get_nested(obj: dict, dotted_path: str):
    cur = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _format_float(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_optional(value) -> str:
    return "" if value is None else str(value)


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
        task_config = TASK_CONFIGS.get(path.stem)
        result = evaluate_audio(
            path,
            args.output_dir,
            args.checkpoint,
            args.device,
            task_config,
        )
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
