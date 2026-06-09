#!/usr/bin/env python3
"""Check behavior-level regression expectations for self-recorded WAV samples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_audio import evaluate_audio  # noqa: E402
from scripts.eval.evaluate_self_recorded import TASK_CONFIGS  # noqa: E402


EXPECTED_SAMPLES = [
    "00_silence",
    "01_speaking_voice",
    "03_sustained_aaa",
    "04_pitch_slide",
    "05_twinkle_twinkle",
]

INVALID_NO_VOICE_TYPES = {"no_voice_or_noise", "invalid", "no_voice", "noise"}
SPEECH_TYPES = {"speech_like_or_non_singing", "speech", "non_singing_voice"}
SUSTAINED_TYPES = {"sustained_note", "diagnostic_sustained_tone"}
PITCH_SLIDE_TYPES = {"pitch_slide", "diagnostic_pitch_slide"}
FREE_SINGING_TYPES = {"free_singing", "analyzable_singing"}


@dataclass
class CheckResult:
    sample: str
    passed: bool
    reasons: list[str]
    artifact: Path | None = None


def _load_or_run(
    sample: str,
    output_dir: Path,
    samples_dir: Path,
    checkpoint: Path,
    device: str,
) -> tuple[dict[str, Any], Path]:
    artifact = output_dir / sample / f"{sample}.json"
    if artifact.exists():
        data = json.loads(artifact.read_text(encoding="utf-8"))
        if data.get("status") == "success":
            return data, artifact

    audio = samples_dir / f"{sample}.wav"
    if not audio.exists():
        raise FileNotFoundError(f"Missing evaluation artifact and WAV sample: {artifact}, {audio}")

    result = evaluate_audio(
        audio,
        output_dir,
        checkpoint,
        device,
        TASK_CONFIGS.get(sample),
    )
    if result.get("status") != "success":
        raise RuntimeError(f"Evaluation failed for {sample}: {result.get('error')}")
    return result, artifact


def _metrics(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("summary_metrics") or {}


def _result(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("result") or {}


def _validity(data: dict[str, Any]) -> dict[str, Any]:
    return (_metrics(data).get("analysis_validity") or _result(data).get("analysis_validity") or {})


def _task_config(data: dict[str, Any]) -> dict[str, Any]:
    return (_metrics(data).get("task_config") or _result(data).get("task_config") or {})


def _task_analysis(data: dict[str, Any]) -> dict[str, Any]:
    return (_metrics(data).get("task_analysis") or _result(data).get("task_analysis") or {})


def _input_type(data: dict[str, Any]) -> str:
    return str(_validity(data).get("input_type") or "")


def _task_type(data: dict[str, Any]) -> str:
    return str(_task_config(data).get("task_type") or _task_analysis(data).get("task_type") or "")


def _score_status(data: dict[str, Any]) -> str:
    return str(_metrics(data).get("score_status") or _result(data).get("score_status") or "")


def _full_song_score(data: dict[str, Any]) -> Any:
    if "full_song_score" in _metrics(data):
        return _metrics(data).get("full_song_score")
    return _result(data).get("full_song_score")


def _diagnostic_score_present(data: dict[str, Any]) -> bool:
    return "diagnostic_score" in _metrics(data) or "diagnostic_score" in _result(data)


def _diagnostic_score(data: dict[str, Any]) -> Any:
    if "diagnostic_score" in _metrics(data):
        return _metrics(data).get("diagnostic_score")
    return _result(data).get("diagnostic_score")


def _issues(data: dict[str, Any]) -> list[Any]:
    return list(_result(data).get("issues") or [])


def _exercises(data: dict[str, Any]) -> list[Any]:
    return list(_result(data).get("exercises") or [])


def _note_count(data: dict[str, Any]) -> int:
    value = _metrics(data).get("note_count")
    if value is not None:
        return int(value)
    return len(_result(data).get("notes") or [])


def _text_blob(data: dict[str, Any]) -> str:
    parts: list[str] = []
    result = _result(data)
    task = _task_analysis(data)
    for key in ("summary", "score_caveat", "score_status"):
        if result.get(key):
            parts.append(str(result[key]))
    for key in ("summary", "status"):
        if task.get(key):
            parts.append(str(task[key]))
    parts.extend(str(item) for item in task.get("caveats") or [])
    parts.extend(str(item) for item in _issues(data))
    parts.extend(str(item) for item in _exercises(data))
    return " ".join(parts).lower()


def _check(condition: bool, reasons: list[str], ok: str, fail: str) -> None:
    reasons.append(("PASS: " if condition else "FAIL: ") + (ok if condition else fail))


def check_00_silence(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    input_type = _input_type(data)
    validity = _validity(data)
    _check(
        input_type in INVALID_NO_VOICE_TYPES,
        reasons,
        f"input_type is {input_type}",
        f"input_type is {input_type}, expected no_voice_or_noise/equivalent",
    )
    _check(_full_song_score(data) is None, reasons, "full_song_score is null", "full_song_score is not null")
    _check(_diagnostic_score(data) is None, reasons, "diagnostic_score is null", "diagnostic_score is not null")
    _check(not _issues(data), reasons, "no user-facing issues", f"issues present: {_issues(data)}")
    _check(not _exercises(data), reasons, "no user-facing exercises", f"exercises present: {_exercises(data)}")
    _check(
        input_type != "analyzable_singing" and not bool(validity.get("is_analyzable")),
        reasons,
        "not analyzable_singing",
        "silence/noise was treated as analyzable singing",
    )
    return reasons


def check_01_speaking_voice(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    input_type = _input_type(data)
    validity = _validity(data)
    _check(
        input_type in SPEECH_TYPES,
        reasons,
        f"input_type is {input_type}",
        f"input_type is {input_type}, expected speech_like_or_non_singing/equivalent",
    )
    _check(_full_song_score(data) is None, reasons, "full_song_score is null", "full_song_score is not null")
    _check(not _exercises(data), reasons, "no singing exercises", f"exercises present: {_exercises(data)}")
    _check(
        input_type != "analyzable_singing" and not bool(validity.get("is_analyzable")),
        reasons,
        "not analyzable_singing",
        "speech was treated as analyzable singing",
    )
    return reasons


def check_03_sustained_aaa(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    task_or_input = {_task_type(data), _input_type(data), _score_status(data)}
    _check(
        bool(task_or_input & SUSTAINED_TYPES) or "diagnostic_sustained_tone_only" in task_or_input,
        reasons,
        f"sustained diagnostic mode observed: {sorted(task_or_input)}",
        f"sustained diagnostic mode not observed: {sorted(task_or_input)}",
    )
    _check(_full_song_score(data) is None, reasons, "full_song_score is null", "full_song_score is not null")
    _check(_diagnostic_score_present(data), reasons, "diagnostic_score field exists", "diagnostic_score field missing")
    _check(_diagnostic_score(data) is not None, reasons, "diagnostic_score has a value", "diagnostic_score is null")
    _check(_note_count(data) <= 5, reasons, f"coaching note count is {_note_count(data)}", f"coaching note count is excessive: {_note_count(data)}")
    _check(
        "full_song" not in _score_status(data) and _task_type(data) == "sustained_note",
        reasons,
        "not treated as full-song melody",
        f"unexpected full-song treatment: task={_task_type(data)} status={_score_status(data)}",
    )
    return reasons


def check_04_pitch_slide(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    task_or_input = {_task_type(data), _input_type(data), _score_status(data)}
    _check(
        bool(task_or_input & PITCH_SLIDE_TYPES) or "diagnostic_pitch_slide_only" in task_or_input,
        reasons,
        f"pitch-slide diagnostic mode observed: {sorted(task_or_input)}",
        f"pitch-slide diagnostic mode not observed: {sorted(task_or_input)}",
    )
    _check(_full_song_score(data) is None, reasons, "full_song_score is null", "full_song_score is not null")
    _check(_diagnostic_score_present(data), reasons, "diagnostic_score field exists", "diagnostic_score field missing")
    _check(_diagnostic_score(data) is not None, reasons, "diagnostic_score has a value", "diagnostic_score is null")
    _check(
        "excellent singing" not in _text_blob(data),
        reasons,
        "no generic 'excellent singing' praise",
        "generic full-song praise found",
    )
    return reasons


def check_05_twinkle_twinkle(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    task_or_input = {_task_type(data), _input_type(data)}
    text = _text_blob(data)
    _check(
        bool(task_or_input & FREE_SINGING_TYPES),
        reasons,
        f"free/analyzable singing observed: {sorted(task_or_input)}",
        f"expected free_singing or analyzable_singing, got {sorted(task_or_input)}",
    )
    _check(
        _full_song_score(data) is None or isinstance(_full_song_score(data), (int, float)),
        reasons,
        "full_song_score is absent or numeric",
        f"full_song_score has unexpected type/value: {_full_song_score(data)!r}",
    )
    _check(
        "reference melody" in text or "not reference" in text or "no reference" in text,
        reasons,
        "reference-melody caveat present",
        "reference-melody caveat missing",
    )
    return reasons


CHECKS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "00_silence": check_00_silence,
    "01_speaking_voice": check_01_speaking_voice,
    "03_sustained_aaa": check_03_sustained_aaa,
    "04_pitch_slide": check_04_pitch_slide,
    "05_twinkle_twinkle": check_05_twinkle_twinkle,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eval/self_recorded"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[CheckResult] = []
    for sample in EXPECTED_SAMPLES:
        try:
            data, artifact = _load_or_run(
                sample,
                args.output_dir,
                args.samples_dir,
                args.checkpoint,
                args.device,
            )
            reasons = CHECKS[sample](data)
            passed = all(reason.startswith("PASS:") for reason in reasons)
            results.append(CheckResult(sample=sample, passed=passed, reasons=reasons, artifact=artifact))
        except Exception as exc:  # noqa: BLE001 - surfaced as regression failure.
            results.append(CheckResult(sample=sample, passed=False, reasons=[f"FAIL: {exc}"]))

    for item in results:
        status = "PASS" if item.passed else "FAIL"
        artifact = f" ({item.artifact})" if item.artifact else ""
        print(f"{status} {item.sample}{artifact}")
        for reason in item.reasons:
            print(f"  - {reason}")

    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
