#!/usr/bin/env python3
"""Run H3 task-specific evaluators on self-recorded WAV samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import task_evaluators as h3  # noqa: E402
from scripts.eval import ui_ready_analysis as h2  # noqa: E402


TASK_CONFIGS = {
    "00_silence": {
        "task_type": "free_singing",
        "skill_focus": "invalid_input_regression",
        "target": None,
        "reference": None,
        "strictness": "beginner",
        "expected_duration": None,
        "expected_direction": None,
    },
    "01_speaking_voice": {
        "task_type": "free_singing",
        "skill_focus": "invalid_input_regression",
        "target": None,
        "reference": None,
        "strictness": "beginner",
        "expected_duration": None,
        "expected_direction": None,
    },
    "03_sustained_aaa": {
        "task_type": "sustained_note",
        "skill_focus": ["pitch_stability", "voiced_continuity"],
        "target": {"vowel": "aa"},
        "reference": None,
        "strictness": "beginner",
        "expected_duration": {"min_s": 4.0, "target_s": 5.0, "max_s": 8.0},
        "expected_direction": "flat",
    },
    "04_pitch_slide": {
        "task_type": "pitch_slide",
        "skill_focus": ["direction", "smoothness"],
        "target": {"direction": "up_or_down"},
        "reference": None,
        "strictness": "beginner",
        "expected_duration": {"min_s": 3.0, "target_s": 5.0, "max_s": 8.0},
        "expected_direction": "up_or_down",
    },
    "05_twinkle_twinkle": {
        "task_type": "free_singing",
        "skill_focus": ["general_pitch_contour", "phrase_continuity"],
        "target": None,
        "reference": None,
        "strictness": "beginner",
        "expected_duration": None,
        "expected_direction": None,
    },
}


def apply_h3(analysis: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
    analysis = dict(analysis)
    analysis["task_config"] = task_config
    evaluated = h3.evaluate_task(analysis, task_config)
    analysis["task_result"] = evaluated["task_result"]
    analysis["subscores"] = evaluated["subscores"]
    analysis["feedback_policy"] = evaluated["feedback_policy"]
    analysis["h3_task_evaluator"] = {
        "allowed_feedback": evaluated["allowed_feedback"],
        "blocked_feedback": evaluated["blocked_feedback"],
        "caveats": evaluated["caveats"],
        "next_exercise_suggestion": evaluated["next_exercise_suggestion"],
    }
    return analysis


def summarize(analysis: dict[str, Any]) -> dict[str, Any]:
    task_result = analysis["task_result"]
    feedback = analysis["feedback_policy"]
    subscores = analysis.get("subscores") or {}
    return {
        "sample": Path(analysis["input_path"]).stem,
        "task_type": task_result.get("task_type"),
        "status": task_result.get("status"),
        "full_song_score": task_result.get("full_song_score"),
        "diagnostic_score": task_result.get("diagnostic_score"),
        "allowed_feedback": feedback.get("allowed_feedback"),
        "blocked_feedback_count": len(feedback.get("blocked_feedback") or []),
        "caveats": feedback.get("caveats"),
        "next_exercise_suggestion": task_result.get("next_exercise_suggestion"),
        "subscores": subscores,
    }


def compact_json(value: Any, max_chars: int = 2200) -> str:
    text = json.dumps(value, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n  ...\n}"


def write_report(path: Path, outputs: list[dict[str, Any]], placeholders: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# H3 Task-Specific Evaluator Report",
        "",
        "First task-specific evaluators for H2 UI-ready analysis JSON.",
        "",
        "No frontend, retraining, model architecture, or existing scoring/API behavior was changed.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        "- Per-sample JSON files are saved as `reports/task_evaluators/<sample>/<sample>_h3_task_analysis.json`.",
        "",
        "## Self-Recorded Results",
        "",
        "| Sample | Task | Status | Full-song score | Diagnostic score | Allowed feedback | Caveats | Next exercise |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    summaries = [summarize(item) for item in outputs]
    for item in summaries:
        next_ex = item["next_exercise_suggestion"] or {}
        lines.append(
            "| {sample} | `{task}` | `{status}` | {full} | {diag} | `{allowed}` | {caveats} | {next_title} |".format(
                sample=f"`{item['sample']}`",
                task=item["task_type"],
                status=item["status"],
                full=item["full_song_score"] if item["full_song_score"] is not None else "null",
                diag=item["diagnostic_score"] if item["diagnostic_score"] is not None else "null",
                allowed=", ".join(item["allowed_feedback"] or []),
                caveats=", ".join(item["caveats"] or []).replace("|", "\\|"),
                next_title=(next_ex.get("title") or "").replace("|", "\\|"),
            )
        )

    lines += [
        "",
        "## Implemented Evaluators",
        "",
        "- `sustained_note`: pitch stability, drift, voiced continuity, dropout rate, volume steadiness.",
        "- `pitch_slide`: direction, range, smoothness, continuity, dropout rate.",
        "- `free_singing`: general pitch stability, phrase continuity, signal quality, no-reference caveat.",
        "- `note_match`: requires `target.note` or `target.f0_hz`; returns `insufficient_target_info` otherwise.",
        "- `reference_song`: computes provisional f0-contour feedback when note f0/durations are supplied; still blocks full-song/rhythm accuracy.",
        "",
        "## Placeholder Checks",
        "",
    ]
    for item in placeholders:
        lines += [
            f"### `{item['task_result']['task_type']}`",
            "",
            "```json",
            json.dumps(
                {
                    "task_result": item["task_result"],
                    "subscores": item["subscores"],
                    "feedback_policy": item["feedback_policy"],
                },
                indent=2,
            ),
            "```",
            "",
        ]

    lines += [
        "## Example Subscores",
        "",
    ]
    for item in summaries:
        lines += [
            f"### `{item['sample']}`",
            "",
            "```json",
            compact_json(item["subscores"]),
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/task_evaluators"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=h2.h1b.h1.NANOPITCH_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for path in h2.SELF_RECORDED:
        sample = path.stem
        task_config = TASK_CONFIGS[sample]
        print(f"Evaluating H3 task {task_config['task_type']} for {path}")
        analysis = h2.build_ui_ready_analysis(
            path,
            checkpoint=args.checkpoint,
            nanopitch_checkpoint=args.nanopitch_checkpoint,
            device=args.device,
            debug=not args.no_debug,
        )
        analysis = apply_h3(analysis, task_config)
        sample_dir = args.output_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"{sample}_h3_task_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        outputs.append(analysis)

    # Prove placeholder behavior without needing extra audio runs.
    base = outputs[-1]
    placeholders = [
        h3.evaluate_task(dict(base, task_config={"task_type": "note_match", "target": None}), {"task_type": "note_match", "target": None}),
        h3.evaluate_task(
            dict(base, task_config={"task_type": "reference_song", "reference": None}),
            {"task_type": "reference_song", "reference": None},
        ),
    ]
    placeholder_dir = args.output_dir / "_placeholder_checks"
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    (placeholder_dir / "note_match_missing_target.json").write_text(json.dumps(placeholders[0], indent=2), encoding="utf-8")
    (placeholder_dir / "reference_song_missing_reference.json").write_text(json.dumps(placeholders[1], indent=2), encoding="utf-8")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps([summarize(item) for item in outputs], indent=2), encoding="utf-8")
    write_report(Path("H3_TASK_EVALUATOR_REPORT.md"), outputs, placeholders, args.output_dir)
    print(json.dumps({"status": "complete", "summary": str(summary_path), "report": "H3_TASK_EVALUATOR_REPORT.md"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
