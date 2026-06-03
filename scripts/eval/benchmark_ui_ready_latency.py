#!/usr/bin/env python3
"""Benchmark UI-ready endpoint latency on the current singing-coach fixtures."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app  # noqa: E402


CASES: list[dict[str, Any]] = [
    {
        "sample": "00_silence",
        "path": REPO_ROOT / "samples" / "00_silence.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
    },
    {
        "sample": "01_speaking_voice",
        "path": REPO_ROOT / "samples" / "01_speaking_voice.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
    },
    {
        "sample": "03_sustained_aaa",
        "path": REPO_ROOT / "samples" / "03_sustained_aaa.wav",
        "task_config": {
            "task_type": "sustained_note",
            "target": {"note": "C4", "f0_hz": 261.63},
            "expected_duration": 5,
            "strictness": "beginner",
        },
    },
    {
        "sample": "04_pitch_slide",
        "path": REPO_ROOT / "samples" / "04_pitch_slide.wav",
        "task_config": {
            "task_type": "pitch_slide",
            "expected_direction": "up",
            "expected_duration": 5,
            "strictness": "beginner",
        },
    },
    {
        "sample": "05_twinkle_twinkle",
        "path": REPO_ROOT / "samples" / "05_twinkle_twinkle.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
    },
    {
        "sample": "05_twinkle_reference",
        "path": REPO_ROOT / "samples" / "05_twinkle_twinkle.wav",
        "task_config": {
            "task_type": "reference_song",
            "reference": {
                "type": "midi_note_sequence",
                "title": "Twinkle opening latency benchmark",
                "notes": ["C4", "C4", "G4", "G4", "A4", "A4", "G4"],
                "f0_hz": [261.63, 261.63, 392.0, 392.0, 440.0, 440.0, 392.0],
                "durations_s": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            },
            "strictness": "beginner",
        },
    },
]


def post_case(client: TestClient, case: dict[str, Any], include_frames: bool) -> dict[str, Any]:
    path = Path(case["path"])
    start = time.perf_counter()
    with path.open("rb") as fh:
        response = client.post(
            "/api/audio/analyze-with-ml?response_mode=ui_ready",
            files={"file": (path.name, fh, "audio/wav")},
            data={
                "song_title": case["sample"],
                "artist": "Latency Benchmark",
                "task_config": json.dumps(case["task_config"]),
                "include_frames": "true" if include_frames else "false",
                "debug": "false",
            },
        )
    client_elapsed = time.perf_counter() - start
    body = response.json()
    if response.status_code != 200 or body.get("status") != "success":
        return {
            "sample": case["sample"],
            "include_frames": include_frames,
            "ok": False,
            "http_status": response.status_code,
            "client_elapsed_s": round(client_elapsed, 4),
            "error": body.get("error") or body,
        }
    analysis = body.get("uiReadyAnalysis") or {}
    perf = analysis.get("performance") or {}
    audio = analysis.get("audio") or {}
    duration = float(audio.get("duration_s") or perf.get("audio_duration_s") or 0.0)
    endpoint_total = perf.get("endpoint_total_s")
    realtime_factor = perf.get("ui_ready_realtime_factor")
    if isinstance(endpoint_total, (int, float)) and duration > 0:
        endpoint_realtime_factor = float(endpoint_total) / duration
    else:
        endpoint_realtime_factor = None
    return {
        "sample": case["sample"],
        "include_frames": include_frames,
        "ok": True,
        "input_type": (analysis.get("analysis_validity") or {}).get("input_type"),
        "task_status": (analysis.get("task_result") or {}).get("status"),
        "audio_duration_s": duration,
        "client_elapsed_s": round(client_elapsed, 4),
        "performance": perf,
        "ui_ready_realtime_factor": realtime_factor,
        "endpoint_realtime_factor": round(endpoint_realtime_factor, 4) if endpoint_realtime_factor is not None else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in records if item.get("ok")]
    client_times = [float(item["client_elapsed_s"]) for item in successful]
    endpoint_factors = [
        float(item["endpoint_realtime_factor"])
        for item in successful
        if isinstance(item.get("endpoint_realtime_factor"), (int, float))
    ]
    return {
        "record_count": len(records),
        "success_count": len(successful),
        "client_elapsed_s_median": round(statistics.median(client_times), 4) if client_times else None,
        "client_elapsed_s_max": round(max(client_times), 4) if client_times else None,
        "endpoint_realtime_factor_median": round(statistics.median(endpoint_factors), 4) if endpoint_factors else None,
        "endpoint_realtime_factor_max": round(max(endpoint_factors), 4) if endpoint_factors else None,
        "realtime_note": "factor < 1.0 is faster than audio duration for completed-file analysis; live streaming still requires a separate browser/on-device design.",
    }


def write_markdown(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# UI-Ready Latency Benchmark",
        "",
        "Report-only benchmark for the current `/api/audio/analyze-with-ml?response_mode=ui_ready` path.",
        "",
        "This measures completed-file backend analysis using FastAPI TestClient. It is not a browser streaming benchmark.",
        "",
        "## Summary",
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Successes: `{summary['success_count']}`",
        f"- Median client elapsed: `{summary['client_elapsed_s_median']}` s",
        f"- Max client elapsed: `{summary['client_elapsed_s_max']}` s",
        f"- Median endpoint realtime factor: `{summary['endpoint_realtime_factor_median']}`",
        f"- Max endpoint realtime factor: `{summary['endpoint_realtime_factor_max']}`",
        "",
        "## Results",
        "",
        "| Sample | Frames | Input type | Task status | Audio s | Client s | Endpoint realtime factor | Checkpoint s | UI-ready s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in records:
        perf = item.get("performance") or {}
        lines.append(
            "| {sample} | {frames} | `{input_type}` | `{status}` | {audio:.2f} | {client:.2f} | {factor} | {checkpoint} | {ui_ready} |".format(
                sample=item["sample"],
                frames="yes" if item["include_frames"] else "no",
                input_type=item.get("input_type") or "error",
                status=item.get("task_status") or item.get("error") or "error",
                audio=float(item.get("audio_duration_s") or 0.0),
                client=float(item.get("client_elapsed_s") or 0.0),
                factor=item.get("endpoint_realtime_factor"),
                checkpoint=perf.get("checkpoint_coaching_s"),
                ui_ready=perf.get("build_ui_ready_response_s"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `endpoint_realtime_factor < 1.0` means completed-file backend analysis was faster than the clip duration.",
            "- This does not prove live browser real-time inference; that still needs a streaming/on-device architecture and browser mic verification.",
            "- Use this benchmark as a regression guard while model, visualization, and hybrid-source work changes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = REPO_ROOT / "reports" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    records = []
    for include_frames in (False, True):
        for case in CASES:
            records.append(post_case(client, case, include_frames=include_frames))
            latest = records[-1]
            print(
                "PASS" if latest.get("ok") else "FAIL",
                latest["sample"],
                "frames" if include_frames else "no_frames",
                latest.get("client_elapsed_s"),
                latest.get("endpoint_realtime_factor"),
            )
    summary = summarize(records)
    payload = {
        "schema_version": "ui_ready_latency_benchmark.v1",
        "summary": summary,
        "records": records,
    }
    (output_dir / "ui_ready_latency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(output_dir / "UI_READY_LATENCY.md", records, summary)
    return 0 if summary["success_count"] == summary["record_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
