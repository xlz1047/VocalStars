#!/usr/bin/env python3
"""Create a consolidated report comparing available model-stack outputs.

This script does not train or modify product behavior. It consolidates the
evaluation evidence that can be computed today:

- self-recorded WAV task behavior and Model C prediction inspection.
- Model A / NanoPitch / pyin / hybrid VAD-f0 metrics on annotated MIR-1K.
- Model A vs Model C rough technique checks on labelled GTsinger/VocalSet clips.

Model C is not scored on pitch/VAD because it is not a pitch/VAD model. It is
not scored on MIR-1K technique because MIR-1K does not provide GTsinger-style
technique labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SELF_WAVS = [
    Path("samples/00_silence.wav"),
    Path("samples/01_speaking_voice.wav"),
    Path("samples/03_sustained_aaa.wav"),
    Path("samples/04_pitch_slide.wav"),
    Path("samples/05_twinkle_twinkle.wav"),
]


def run_cmd(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def maybe_run(args: argparse.Namespace, commands: list[dict[str, Any]]) -> None:
    if args.skip_runs:
        return
    py = args.python
    commands.append(
        run_cmd(
            [
                py,
                "scripts/eval/hybrid_decision_harness.py",
                "--device",
                args.device,
                "--files",
                *[str(path) for path in SELF_WAVS],
            ],
            REPO_ROOT,
        )
    )
    commands.append(
        run_cmd(
            [
                py,
                "scripts/eval/evaluate_task_specific.py",
                "--device",
                args.device,
            ],
            REPO_ROOT,
        )
    )
    commands.append(
        run_cmd(
            [
                py,
                "scripts/eval/compare_model_a_model_c.py",
                "--device",
                args.device,
                "--clips-per-label",
                str(args.clips_per_label),
            ],
            REPO_ROOT,
        )
    )
    if args.run_mir1k:
        commands.append(
            run_cmd(
                [
                    py,
                    "scripts/eval/evaluate_human_singing_manifest.py",
                    "--device",
                    args.device,
                    "--limit",
                    str(args.mir1k_limit),
                    "--detail-limit",
                    str(min(args.mir1k_limit, 10)),
                ],
                REPO_ROOT,
            )
        )
    commands.append(run_cmd([py, "scripts/eval/check_regression_expectations.py"], REPO_ROOT))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def task_summary() -> list[dict[str, Any]]:
    return read_json(REPO_ROOT / "reports/task_evaluators/summary.json", [])


def hybrid_self_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in SELF_WAVS:
        stem = path.stem
        data = read_json(REPO_ROOT / "reports/hybrid_decision" / stem / f"{stem}_hybrid_decision.json", {})
        if not data:
            continue
        rows.append(
            {
                "sample": stem,
                "recommendation": data.get("recommendation", {}),
                "sources": {
                    source: {
                        "voiced_percentage": (data.get("sources", {}).get(source, {}).get("voiced_percentage")),
                        "f0_coverage": (data.get("sources", {}).get(source, {}).get("f0_coverage")),
                        "median_f0_hz": (data.get("sources", {}).get(source, {}).get("median_f0_hz")),
                    }
                    for source in ("model_a", "nanopitch", "pyin")
                },
            }
        )
    return rows


def mir1k_summary() -> dict[str, Any]:
    data = read_json(REPO_ROOT / "reports/human_singing_eval/mir1k/summary.json", {})
    return data.get("aggregate", {}) if data else {}


def model_compare_summary() -> dict[str, Any]:
    return read_json(REPO_ROOT / "reports/model_comparison/model_a_vs_model_c/summary.json", {})


def model_c_heldout_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("test_alto1", "test_tenor1", "test_vocalset"):
        data = read_json(REPO_ROOT / "reports/model_c/run_alto2_v3" / f"{split}_evaluation.json", {})
        if not data:
            continue
        out[split] = {
            "passes_gate": data.get("deployment_flags", {}).get("passes_report_only_gate"),
            "dominant_prediction": data.get("evaluation", {}).get("dominant_prediction"),
            "per_technique": {
                tech: {
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "false_positive_rate": m.get("false_positive_rate"),
                }
                for tech, m in (data.get("evaluation", {}).get("per_technique") or {}).items()
                if tech in {"falsetto", "breathy", "pharyngeal", "glissando", "vibrato"}
            },
        }
    return out


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def cents(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Model Stack Comparison",
        "",
        "This report compares the current approaches only on outputs where labels or task expectations are available.",
        "",
        "## What Each Approach Can Do",
        "",
        "| Approach | Pitch/f0 | VAD/no-voice | Technique labels | Current role |",
        "|---|---|---|---|---|",
        "| Model A unified | yes | yes, but over-voices noise/speech | 20-class clip head, unreliable | broad production coverage with gates |",
        "| NanoPitch | yes, sparse/conservative | strong conservative guard | no | no-voice/noise guard and browser candidate |",
        "| pyin | yes | not a singing-validity classifier | no | diagnostic f0 reference when stable |",
        "| Hybrid | selected from above | selected/guarded from above | no | safest current production pitch/VAD strategy |",
        "| Model C v3 | no direct pitch/VAD output; uses NanoPitch features | no direct validity head | six frame-level heads, `mix` hidden | report-only technique research |",
        "",
        "## Self-Recorded WAVs",
        "",
        "These samples have task expectations but no technique labels. Technique accuracy cannot be computed; we can only inspect whether predictions look safe.",
        "",
        "| Sample | Task status | Score | F0 source recommendation | VAD recommendation | Model A voiced | NanoPitch voiced | pyin voiced |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    tasks = {item.get("sample"): item for item in payload["task_summary"]}
    for item in payload["hybrid_self_summary"]:
        sample = item["sample"]
        task = tasks.get(sample, {})
        rec = item["recommendation"]
        sources = item["sources"]
        score = task.get("diagnostic_score")
        if score is None:
            score = task.get("full_song_score")
        lines.append(
            "| {sample} | `{status}` | {score} | `{f0}` | `{vad}` | {ma} | {np} | {pyin} |".format(
                sample=f"`{sample}`",
                status=task.get("status"),
                score="null" if score is None else score,
                f0=rec.get("selected_f0_source_recommendation"),
                vad=rec.get("selected_vad_source_recommendation"),
                ma=pct(sources["model_a"]["voiced_percentage"]),
                np=pct(sources["nanopitch"]["voiced_percentage"]),
                pyin=pct(sources["pyin"]["voiced_percentage"]),
            )
        )

    comp = payload["model_a_vs_model_c"]
    lines += [
        "",
        "## Same Self WAVs: Technique Prediction Inspection",
        "",
        "No accuracy is claimed here because these WAVs do not have technique labels.",
        "",
        "| Sample | Model A top technique | Model C top visible label | Model C predicted labels | Interpretation |",
        "|---|---|---|---|---|",
    ]
    for item in comp.get("self_wav_predictions", []):
        ma = item["model_a"]
        mc = item["model_c"]
        sample = item["sample"]
        interpretation = "unsafe_for_user_facing_technique"
        if sample == "00_silence" and not mc.get("predicted_user_facing_labels_at_5pct_frames"):
            interpretation = "Model C abstains enough here, but validity should still come from NanoPitch/task gate"
        lines.append(
            "| `{sample}` | `{a}` | `{c}` ({frac}) | `{preds}` | {interp} |".format(
                sample=sample,
                a=ma.get("top_label"),
                c=mc.get("top_user_facing_label"),
                frac=pct(mc.get("top_user_facing_predicted_fraction")),
                preds=", ".join(mc.get("predicted_user_facing_labels_at_5pct_frames") or []),
                interp=interpretation,
            )
        )

    lines += [
        "",
        "## MIR-1K VAD/f0 Accuracy",
        "",
        "MIR-1K has f0/voicing labels, so it can compare Model A, NanoPitch, pyin, and hybrid. Model C is not included because it has no pitch/VAD head.",
        "",
        "| Source | Voiced F1 | False voiced | F0 coverage | Median f0 error cents | Mean f0 error cents |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    mir = payload["mir1k_summary"]
    for source, metrics in (mir.get("sources") or {}).items():
        lines.append(
            "| `{source}` | {f1} | {fv} | {cov} | {med} | {mean} |".format(
                source=source,
                f1=pct(metrics.get("voiced_f1")),
                fv=pct(metrics.get("false_voiced_rate")),
                cov=pct(metrics.get("f0_coverage_on_ref_voiced")),
                med=cents(metrics.get("median_abs_f0_error_cents")),
                mean=cents(metrics.get("mean_abs_f0_error_cents")),
            )
        )

    lines += [
        "",
        "## Technique Accuracy Where Labels Exist",
        "",
        "This is the only place where Model C can be scored directly. It uses GTsinger/VocalSet labels that overlap Model A and Model C.",
        "",
        "| Split | Model A rough top-1 match | Model C rough detect rate | Decision |",
        "|---|---:|---:|---|",
    ]
    for split, summary in (comp.get("aggregate") or {}).items():
        lines.append(
            "| `{split}` | {ma} | {mc} | {decision} |".format(
                split=split,
                ma=num(summary.get("model_a_top1_match_rate_on_comparable")),
                mc=num(summary.get("model_c_detect_rate_on_comparable")),
                decision="not deployable",
            )
        )

    lines += [
        "",
        "## Model C v3 Held-Out Gates",
        "",
        "| Split | Gate | Main failing reason |",
        "|---|---|---|",
    ]
    for split, summary in payload["model_c_heldout"].items():
        dominant = summary.get("dominant_prediction") or {}
        reason = "non-mix technique precision/recall/FPR not good enough"
        if dominant.get("share") and float(dominant["share"]) > 0.60:
            reason = f"dominant `{dominant.get('technique')}` predictions ({pct(dominant.get('share'))})"
        lines.append(f"| `{split}` | `{summary.get('passes_gate')}` | {reason} |")

    lines += [
        "",
        "## Decision",
        "",
        "- Do not switch production to Model C.",
        "- Keep the current hybrid pitch/VAD path: NanoPitch guard + pyin diagnostic f0 + Model A free-singing coverage.",
        "- Keep Model C v3 report-only for technique research.",
        "- Model C can be evaluated now on arbitrary WAVs for prediction inspection and on labelled GTsinger/VocalSet clips for technique metrics.",
        "- A future Model C integration should be additive only: technique markers behind strict gates after held-out singer metrics pass.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison/model_stack"))
    parser.add_argument("--python", default="ml/.venv/bin/python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clips-per-label", type=int, default=3)
    parser.add_argument("--run-mir1k", action="store_true")
    parser.add_argument("--mir1k-limit", type=int, default=10)
    parser.add_argument("--skip-runs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    maybe_run(args, commands)
    payload = {
        "commands": commands,
        "task_summary": task_summary(),
        "hybrid_self_summary": hybrid_self_summary(),
        "mir1k_summary": mir1k_summary(),
        "model_a_vs_model_c": model_compare_summary(),
        "model_c_heldout": model_c_heldout_summary(),
        "decision": {
            "switch_to_model_c": False,
            "model_c_role": "report_only_technique_research",
            "production_role": "hybrid_model_a_nanopitch_pyin",
        },
    }
    json_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"status": "complete", "summary": str(json_path), "report": str(md_path)}, indent=2))
    failed = [cmd for cmd in commands if cmd["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
