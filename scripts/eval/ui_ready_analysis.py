#!/usr/bin/env python3
"""Export minimum H2 UI-ready analysis JSON.

This eval/debug exporter wraps normalized selected-source frames with the
minimum H0/H2 containers needed for pitch lane, waveform overlay, issue
markers, and safe result cards. It does not change existing API responses.
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

from ml_new.inference.coach_inference import HOP_S  # noqa: E402
from scripts.eval import ui_ready_frames as h1b  # noqa: E402


SELF_RECORDED = h1b.SELF_RECORDED

BREATH_PROXY_CAVEAT = "Breath/phrase metrics are proxy features and do not diagnose breath support."
TONE_PROXY_CAVEAT = "Tone/timbre metrics are proxy features and do not diagnose timbre or technique."


def load_existing_eval(sample: str) -> dict[str, Any]:
    path = REPO_ROOT / "reports" / "eval" / "self_recorded" / sample / f"{sample}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_eval(eval_data: dict[str, Any]) -> dict[str, Any]:
    return eval_data.get("summary_metrics") or {}


def default_analysis_validity(frame_export: dict[str, Any]) -> dict[str, Any]:
    frames = frame_export["frames"]
    voiced_ratio = sum(1 for f in frames if f["voiced"]) / max(len(frames), 1)
    f0_ratio = sum(1 for f in frames if f["f0_hz"] is not None) / max(len(frames), 1)
    return {
        "is_analyzable": voiced_ratio > 0.2 or f0_ratio > 0.2,
        "input_type": "analyzable_singing" if voiced_ratio > 0.2 else "low_confidence_or_unreliable",
        "confidence": None,
        "reason_codes": ["derived_from_h2_frame_export"],
        "summary_metrics": {
            "selected_voiced_frame_ratio": voiced_ratio,
            "selected_f0_coverage": f0_ratio,
        },
    }


def regionize(mask: list[bool], min_len: int = 1) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and ((not value) or i == len(mask) - 1):
            end = i if not value else i + 1
            if end - start >= min_len:
                regions.append((start, end))
            start = None
    return regions


def segment_confidence(frames: list[dict[str, Any]], start: int, end: int, key: str) -> float | None:
    vals = [
        float(frame[key])
        for frame in frames[start:end]
        if frame.get(key) is not None and math.isfinite(float(frame[key]))
    ]
    if not vals:
        return None
    return round(float(np.mean(vals)), 6)


def f0_stats(frames: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    vals = [
        float(frame["f0_hz"])
        for frame in frames[start:end]
        if frame.get("f0_hz") is not None and float(frame["f0_hz"]) > 0
    ]
    if not vals:
        return {
            "median_f0_hz": None,
            "min_f0_hz": None,
            "max_f0_hz": None,
            "stability_cents": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    median = float(np.median(arr))
    if arr.size >= 3:
        cents = 1200.0 * np.log2(arr / max(median, 1e-9))
        stability = float(np.std(cents))
    else:
        stability = None
    return {
        "median_f0_hz": round(median, 6),
        "min_f0_hz": round(float(np.min(arr)), 6),
        "max_f0_hz": round(float(np.max(arr)), 6),
        "stability_cents": round(stability, 6) if stability is not None else None,
    }


def frame_values(frames: list[dict[str, Any]], start: int, end: int, getter: Any) -> np.ndarray:
    vals = []
    for frame in frames[start:end]:
        value = getter(frame)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def rms_db_values(frames: list[dict[str, Any]], start: int, end: int) -> np.ndarray:
    return frame_values(frames, start, end, lambda frame: (frame.get("volume") or {}).get("rms_db"))


def f0_values(frames: list[dict[str, Any]], start: int, end: int) -> np.ndarray:
    return frame_values(frames, start, end, lambda frame: frame.get("f0_hz"))


def spectral_values(frames: list[dict[str, Any]], start: int, end: int, key: str) -> np.ndarray:
    return frame_values(frames, start, end, lambda frame: (frame.get("spectral_tone_proxy") or {}).get(key))


def rounded(value: float | None, precision: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), precision)


def segment_f0_stability_cents(frames: list[dict[str, Any]], start: int, end: int) -> float | None:
    vals = f0_values(frames, start, end)
    vals = vals[vals > 0]
    if vals.size < 3:
        return None
    median = float(np.median(vals))
    cents = 1200.0 * np.log2(vals / max(median, 1e-9))
    return float(np.std(cents))


def rms_decay_db(frames: list[dict[str, Any]], start: int, end: int) -> float | None:
    vals = rms_db_values(frames, start, end)
    if vals.size < 6:
        return None
    chunk = max(2, vals.size // 3)
    early = vals[:chunk]
    late = vals[-chunk:]
    return float(np.mean(late) - np.mean(early))


def late_phrase_f0_instability_cents(frames: list[dict[str, Any]], start: int, end: int) -> float | None:
    length = max(end - start, 0)
    if length < 6:
        return None
    late_start = start + int(length * 2 / 3)
    return segment_f0_stability_cents(frames, late_start, end)


def f0_dropout_count(frames: list[dict[str, Any]], start: int, end: int, min_len: int = 3) -> int:
    mask = [frame.get("f0_hz") is None for frame in frames[start:end]]
    return len(regionize(mask, min_len=min_len))


def tone_proxy_metrics(frames: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    rms_db = rms_db_values(frames, start, end)
    centroid = spectral_values(frames, start, end, "spectral_centroid_hz")
    flatness = spectral_values(frames, start, end, "spectral_flatness")
    rolloff = spectral_values(frames, start, end, "spectral_rolloff_hz")
    low_ratio = spectral_values(frames, start, end, "low_frequency_ratio")
    harmonicity = spectral_values(frames, start, end, "harmonicity_noise_proxy")
    return {
        "is_proxy": True,
        "rms_stability_db": rounded(float(np.std(rms_db)) if rms_db.size >= 3 else None),
        "spectral_centroid_mean_hz": rounded(float(np.mean(centroid)) if centroid.size else None),
        "spectral_centroid_std_hz": rounded(float(np.std(centroid)) if centroid.size >= 3 else None),
        "spectral_flatness_mean": rounded(float(np.mean(flatness)) if flatness.size else None),
        "spectral_flatness_std": rounded(float(np.std(flatness)) if flatness.size >= 3 else None),
        "spectral_rolloff_mean_hz": rounded(float(np.mean(rolloff)) if rolloff.size else None),
        "spectral_rolloff_std_hz": rounded(float(np.std(rolloff)) if rolloff.size >= 3 else None),
        "low_frequency_ratio_mean": rounded(float(np.mean(low_ratio)) if low_ratio.size else None),
        "harmonicity_noise_proxy_mean": rounded(float(np.mean(harmonicity)) if harmonicity.size else None),
        "f0_stability_cents": rounded(segment_f0_stability_cents(frames, start, end)),
        "caveat": TONE_PROXY_CAVEAT,
    }


def breath_phrase_proxy_metrics(frames: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    duration_s = (end - start) * HOP_S
    voiced = [bool(frame.get("voiced")) for frame in frames[start:end]]
    voiced_continuity = sum(1 for item in voiced if item) / max(len(voiced), 1)
    dropout_count = f0_dropout_count(frames, start, end)
    return {
        "is_proxy": True,
        "phrase_duration_s": rounded(duration_s),
        "voiced_continuity": rounded(voiced_continuity),
        "dropout_count": dropout_count,
        "rms_decay_db": rounded(rms_decay_db(frames, start, end)),
        "late_phrase_f0_instability_cents": rounded(late_phrase_f0_instability_cents(frames, start, end)),
        "caveat": BREATH_PROXY_CAVEAT,
    }


def build_note_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions = regionize([frame.get("f0_hz") is not None for frame in frames], min_len=3)
    notes = []
    for idx, (start, end) in enumerate(regions, start=1):
        stats = f0_stats(frames, start, end)
        notes.append(
            {
                "id": f"note_{idx:03d}",
                "type": "note",
                "start_s": round(start * HOP_S, 6),
                "end_s": round(end * HOP_S, 6),
                "duration_s": round((end - start) * HOP_S, 6),
                **stats,
                "voiced_coverage": round(
                    sum(1 for frame in frames[start:end] if frame["voiced"]) / max(end - start, 1),
                    6,
                ),
                "confidence": segment_confidence(frames, start, end, "pitch_confidence"),
                "source": "selected_f0",
                "ui_severity": "info",
                "summary": "Selected f0 region",
                "tone_consistency_proxy": tone_proxy_metrics(frames, start, end),
            }
        )
    return notes


def build_phrase_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions = regionize([bool(frame["voiced"]) for frame in frames], min_len=3)
    source = "selected_vad"
    summary = "Continuous selected-voiced region"
    if not regions:
        regions = regionize([frame.get("f0_hz") is not None for frame in frames], min_len=5)
        source = "selected_f0_fallback"
        summary = "Continuous selected-f0 proxy region"
    phrases = []
    for idx, (start, end) in enumerate(regions, start=1):
        phrases.append(
            {
                "id": f"phrase_{idx:03d}",
                "type": "phrase",
                "start_s": round(start * HOP_S, 6),
                "end_s": round(end * HOP_S, 6),
                "duration_s": round((end - start) * HOP_S, 6),
                "voiced_coverage": round(
                    sum(1 for frame in frames[start:end] if frame["voiced"]) / max(end - start, 1),
                    6,
                ),
                "confidence": segment_confidence(frames, start, end, "voice_confidence"),
                "source": source,
                "ui_severity": "info",
                "summary": summary,
                "breath_phrase_proxy": breath_phrase_proxy_metrics(frames, start, end),
                "tone_consistency_proxy": tone_proxy_metrics(frames, start, end),
            }
        )
    return phrases


def build_dropout_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    f0_present = [frame.get("f0_hz") is not None for frame in frames]
    if not any(f0_present):
        return []
    first = next(i for i, value in enumerate(f0_present) if value)
    last = len(f0_present) - 1 - next(i for i, value in enumerate(reversed(f0_present)) if value)
    regions = regionize([not value for value in f0_present[first : last + 1]], min_len=5)
    dropouts = []
    for idx, (rel_start, rel_end) in enumerate(regions, start=1):
        start = first + rel_start
        end = first + rel_end
        dropouts.append(
            {
                "id": f"dropout_{idx:03d}",
                "type": "dropout",
                "start_s": round(start * HOP_S, 6),
                "end_s": round(end * HOP_S, 6),
                "duration_s": round((end - start) * HOP_S, 6),
                "confidence": None,
                "source": "selected_f0",
                "ui_severity": "warning",
                "summary": "Gap in selected f0 contour",
            }
        )
    return dropouts


def build_flag_regions(frames: list[dict[str, Any]], flag: str, prefix: str, summary: str) -> list[dict[str, Any]]:
    regions = regionize([bool(frame["signal_quality"].get(flag)) for frame in frames], min_len=3)
    output = []
    for idx, (start, end) in enumerate(regions, start=1):
        output.append(
            {
                "id": f"{prefix}_{idx:03d}",
                "type": prefix,
                "start_s": round(start * HOP_S, 6),
                "end_s": round(end * HOP_S, 6),
                "duration_s": round((end - start) * HOP_S, 6),
                "confidence": None,
                "source": "frame_signal_quality",
                "ui_severity": "warning",
                "summary": summary,
            }
        )
    return output


def build_breath_phrase_proxy_regions(
    frames: list[dict[str, Any]],
    phrases: list[dict[str, Any]],
    dropouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for idx, phrase in enumerate(phrases, start=1):
        start = int(round(float(phrase["start_s"]) / HOP_S))
        end = int(round(float(phrase["end_s"]) / HOP_S))
        metrics = breath_phrase_proxy_metrics(frames, start, end)
        warning = (
            (metrics.get("rms_decay_db") is not None and float(metrics["rms_decay_db"]) < -6.0)
            or (metrics.get("late_phrase_f0_instability_cents") is not None and float(metrics["late_phrase_f0_instability_cents"]) > 120.0)
            or int(metrics.get("dropout_count") or 0) > 0
        )
        regions.append(
            {
                "id": f"breath_phrase_proxy_{idx:03d}",
                "type": "breath_phrase_proxy",
                "start_s": phrase["start_s"],
                "end_s": phrase["end_s"],
                "duration_s": phrase["duration_s"],
                "source": "selected_vad_and_volume_proxy",
                "ui_severity": "warning" if warning else "info",
                "summary": "Phrase continuity proxy region",
                "proxy_metrics": metrics,
                "caveats": [BREATH_PROXY_CAVEAT],
            }
        )

    offset = len(regions)
    for idx, dropout in enumerate(dropouts, start=1):
        if float(dropout.get("duration_s") or 0.0) < 0.2:
            continue
        regions.append(
            {
                "id": f"breath_gap_proxy_{offset + idx:03d}",
                "type": "breath_gap_proxy",
                "start_s": dropout["start_s"],
                "end_s": dropout["end_s"],
                "duration_s": dropout["duration_s"],
                "source": "selected_f0_dropout_proxy",
                "ui_severity": "info",
                "summary": "Gap proxy region",
                "proxy_metrics": {
                    "is_proxy": True,
                    "dropout_duration_s": dropout["duration_s"],
                    "caveat": BREATH_PROXY_CAVEAT,
                },
                "caveats": [BREATH_PROXY_CAVEAT],
            }
        )
    return regions


def build_tone_consistency_proxy_regions(frames: list[dict[str, Any]], phrases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for idx, phrase in enumerate(phrases, start=1):
        start = int(round(float(phrase["start_s"]) / HOP_S))
        end = int(round(float(phrase["end_s"]) / HOP_S))
        metrics = tone_proxy_metrics(frames, start, end)
        warning = (
            (metrics.get("rms_stability_db") is not None and float(metrics["rms_stability_db"]) > 8.0)
            or (metrics.get("spectral_centroid_std_hz") is not None and float(metrics["spectral_centroid_std_hz"]) > 1000.0)
            or (metrics.get("f0_stability_cents") is not None and float(metrics["f0_stability_cents"]) > 180.0)
        )
        regions.append(
            {
                "id": f"tone_consistency_proxy_{idx:03d}",
                "type": "tone_consistency_proxy",
                "start_s": phrase["start_s"],
                "end_s": phrase["end_s"],
                "duration_s": phrase["duration_s"],
                "source": "spectral_volume_f0_proxy",
                "ui_severity": "warning" if warning else "info",
                "summary": "Tone consistency proxy region",
                "proxy_metrics": metrics,
                "caveats": [TONE_PROXY_CAVEAT],
            }
        )
    return regions


def build_segments(frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    notes = build_note_segments(frames)
    phrases = build_phrase_segments(frames)
    dropouts = build_dropout_segments(frames)
    return {
        "notes": notes,
        "phrases": phrases,
        "dropouts": dropouts,
        "unstable_pitch_regions": build_flag_regions(
            frames,
            "source_disagreement",
            "unstable_pitch_region",
            "Pitch sources disagree in this region",
        ),
        "low_confidence_regions": build_flag_regions(
            frames,
            "low_confidence",
            "low_confidence_region",
            "Selected source confidence is low in this region",
        ),
        "breath_phrase_proxy_regions": build_breath_phrase_proxy_regions(frames, phrases, dropouts),
        "tone_consistency_proxy_regions": build_tone_consistency_proxy_regions(frames, phrases),
    }


def mean_metric(items: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    vals = []
    for item in items:
        value: Any = item
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    return round(float(np.mean(vals)), 6)


def voice_quality_proxy_from_eval(eval_data: dict[str, Any]) -> dict[str, Any] | None:
    voice_quality = ((eval_data.get("result") or {}).get("voice_quality") or {})
    if not voice_quality:
        return None
    return {
        "is_proxy": True,
        "hnr_db": rounded(voice_quality.get("hnr_db")),
        "jitter_pct": rounded(voice_quality.get("jitter_pct")),
        "shimmer_pct": rounded(voice_quality.get("shimmer_pct")),
        "caveat": "HNR/jitter/shimmer are DSP proxy metrics here, not vocal diagnosis.",
    }


def proxy_metrics_summary(
    frames: list[dict[str, Any]],
    segments: dict[str, list[dict[str, Any]]],
    eval_data: dict[str, Any],
) -> dict[str, Any]:
    phrases = segments.get("phrases") or []
    breath_regions = segments.get("breath_phrase_proxy_regions") or []
    tone_regions = segments.get("tone_consistency_proxy_regions") or []
    frame_tone = [frame.get("spectral_tone_proxy") or {} for frame in frames]
    return {
        "is_proxy": True,
        "breath_phrase": {
            "is_proxy": True,
            "phrase_count": len(phrases),
            "mean_phrase_duration_s": mean_metric(phrases, ("duration_s",)),
            "mean_voiced_continuity": mean_metric(breath_regions, ("proxy_metrics", "voiced_continuity")),
            "total_dropout_count": int(
                sum(int(((region.get("proxy_metrics") or {}).get("dropout_count") or 0)) for region in breath_regions)
            ),
            "mean_rms_decay_db": mean_metric(breath_regions, ("proxy_metrics", "rms_decay_db")),
            "mean_late_phrase_f0_instability_cents": mean_metric(
                breath_regions,
                ("proxy_metrics", "late_phrase_f0_instability_cents"),
            ),
            "caveat": BREATH_PROXY_CAVEAT,
        },
        "tone_consistency": {
            "is_proxy": True,
            "region_count": len(tone_regions),
            "mean_rms_stability_db": mean_metric(tone_regions, ("proxy_metrics", "rms_stability_db")),
            "mean_spectral_centroid_hz": mean_metric(frame_tone, ("spectral_centroid_hz",)),
            "mean_spectral_flatness": mean_metric(frame_tone, ("spectral_flatness",)),
            "mean_spectral_rolloff_hz": mean_metric(frame_tone, ("spectral_rolloff_hz",)),
            "mean_low_frequency_ratio": mean_metric(frame_tone, ("low_frequency_ratio",)),
            "mean_harmonicity_noise_proxy": mean_metric(frame_tone, ("harmonicity_noise_proxy",)),
            "mean_voiced_region_f0_stability_cents": mean_metric(
                tone_regions,
                ("proxy_metrics", "f0_stability_cents"),
            ),
            "caveat": TONE_PROXY_CAVEAT,
        },
        "voice_quality_proxy": voice_quality_proxy_from_eval(eval_data),
        "caveats": [BREATH_PROXY_CAVEAT, TONE_PROXY_CAVEAT],
    }


def task_result_from_eval(metrics: dict[str, Any], frame_export: dict[str, Any]) -> dict[str, Any]:
    task_analysis = metrics.get("task_analysis") or {}
    task_type = (metrics.get("task_config") or frame_export.get("task_config") or {}).get("task_type")
    return {
        "task_type": task_analysis.get("task_type") or task_type,
        "status": task_analysis.get("status") or metrics.get("score_status") or "not_scored",
        "score_status": metrics.get("score_status") or task_analysis.get("status") or "not_scored",
        "full_song_score": metrics.get("full_song_score"),
        "diagnostic_score": metrics.get("diagnostic_score"),
        "summary": task_analysis.get("summary") or "",
        "next_exercise_suggestion": None,
    }


def feedback_policy_from_eval(metrics: dict[str, Any], frame_export: dict[str, Any]) -> dict[str, Any]:
    task_analysis = metrics.get("task_analysis") or {}
    validity = metrics.get("analysis_validity") or default_analysis_validity(frame_export)
    input_type = validity.get("input_type")
    task_type = (task_analysis.get("task_type") or (metrics.get("task_config") or {}).get("task_type") or "")
    caveats = list(task_analysis.get("caveats") or [])
    if metrics.get("score_caveat") and metrics["score_caveat"] not in caveats:
        caveats.append(metrics["score_caveat"])
    for caveat in (BREATH_PROXY_CAVEAT, TONE_PROXY_CAVEAT):
        if caveat not in caveats:
            caveats.append(caveat)

    allowed: list[str]
    blocked = [
        {
            "type": "reference_melody_accuracy",
            "reason": "No reference melody comparison is available in H2.",
        },
        {
            "type": "timbre_diagnosis",
            "reason": "Tone/timbre labels are not reliable enough for coaching.",
        },
        {
            "type": "breath_diagnosis",
            "reason": "Breath output is proxy-only until labeled breath data exists.",
        },
        {
            "type": "breath_support_claims",
            "reason": "Breath/phrase fields are continuity and volume proxies only.",
        },
        {
            "type": "technique_or_strain_diagnosis",
            "reason": "Proxy metrics cannot diagnose technique, strain, or support.",
        },
    ]

    if input_type in {"no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"}:
        allowed = ["signal_quality"]
        blocked.extend(
            [
                {
                    "type": "note_specific_pitch_advice",
                    "reason": "Input is not analyzable singing.",
                },
                {
                    "type": "singing_exercises",
                    "reason": "Input is not analyzable singing.",
                },
                {
                    "type": "full_song_score",
                    "reason": "Full-song scoring is blocked for invalid/non-singing input.",
                },
            ]
        )
    elif task_type == "sustained_note":
        allowed = ["pitch_stability", "voicing_continuity", "signal_quality"]
        blocked.append(
            {
                "type": "full_song_score",
                "reason": "Sustained-note task is diagnostic-only.",
            }
        )
    elif task_type == "pitch_slide":
        allowed = ["pitch_slide_direction", "pitch_slide_smoothness", "voicing_continuity", "signal_quality"]
        blocked.append(
            {
                "type": "full_song_score",
                "reason": "Pitch-slide task is diagnostic-only.",
            }
        )
    else:
        allowed = ["general_pitch_contour", "voicing_continuity", "signal_quality"]

    return {
        "allowed_feedback": allowed,
        "blocked_feedback": blocked,
        "caveats": caveats,
    }


def strip_source_values(frames: list[dict[str, Any]], debug: bool) -> list[dict[str, Any]]:
    if debug:
        return frames
    stripped = []
    for frame in frames:
        item = dict(frame)
        item.pop("source_values", None)
        stripped.append(item)
    return stripped


def build_ui_ready_analysis(
    audio_path: Path,
    *,
    checkpoint: Path = Path("ml_new/checkpoints/unified/best.pt"),
    nanopitch_checkpoint: Path = h1b.h1.NANOPITCH_CHECKPOINT,
    device: str = "cpu",
    debug: bool = True,
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_export = h1b.build_ui_ready_frames(
        audio_path,
        checkpoint=checkpoint,
        nanopitch_checkpoint=nanopitch_checkpoint,
        device=device,
        debug=debug,
        task_config=task_config,
    )
    sample = Path(audio_path).stem
    eval_data = load_existing_eval(sample)
    metrics = metrics_from_eval(eval_data)
    frames = strip_source_values(frame_export["frames"], debug=debug)
    analysis_validity = metrics.get("analysis_validity") or default_analysis_validity(frame_export)
    task_config = task_config or metrics.get("task_config") or frame_export.get("task_config") or {}
    segments = build_segments(frames)
    proxy_metrics = proxy_metrics_summary(frames, segments, eval_data)

    return {
        "schema_version": "h4.proxy_ui_analysis.v1",
        "input_path": frame_export["input_path"],
        "audio": frame_export["audio"],
        "task_config": task_config,
        "analysis_validity": analysis_validity,
        "frames": frames,
        "segments": segments,
        "proxy_metrics": proxy_metrics,
        "task_result": task_result_from_eval(metrics, frame_export),
        "feedback_policy": feedback_policy_from_eval(metrics, frame_export),
        "debug": {
            "source_values_included": debug,
            "source_strategy": frame_export["source_strategy"],
            "hybrid_metrics": frame_export["hybrid_metrics"],
            "source_summaries": frame_export["debug"].get("source_summaries"),
            "existing_eval_json": eval_data.get("artifacts", {}).get("json"),
        },
    }


def summarize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    frames = analysis["frames"]
    segments = analysis["segments"]
    return {
        "sample": Path(analysis["input_path"]).stem,
        "task_type": analysis["task_config"].get("task_type"),
        "input_type": analysis["analysis_validity"].get("input_type"),
        "frames": len(frames),
        "selected_voiced_percentage": round(
            100.0 * sum(1 for f in frames if f["voiced"]) / max(len(frames), 1),
            2,
        ),
        "selected_f0_frame_count": sum(1 for f in frames if f["f0_hz"] is not None),
        "notes": len(segments["notes"]),
        "phrases": len(segments["phrases"]),
        "dropouts": len(segments["dropouts"]),
        "unstable_pitch_regions": len(segments["unstable_pitch_regions"]),
        "low_confidence_regions": len(segments["low_confidence_regions"]),
        "breath_phrase_proxy_regions": len(segments.get("breath_phrase_proxy_regions") or []),
        "tone_consistency_proxy_regions": len(segments.get("tone_consistency_proxy_regions") or []),
        "task_status": analysis["task_result"].get("status"),
        "allowed_feedback": analysis["feedback_policy"].get("allowed_feedback"),
    }


def compact_example(analysis: dict[str, Any]) -> dict[str, Any]:
    frames = analysis["frames"]
    chosen = None
    for frame in frames:
        if frame.get("f0_hz") is not None and not frame["signal_quality"].get("near_silence"):
            chosen = frame
            break
    if chosen is None:
        for frame in frames:
            if not frame["signal_quality"].get("near_silence"):
                chosen = frame
                break
    if chosen is None and frames:
        chosen = frames[0]
    frame_keys = [
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
    return {
        "task_config": analysis["task_config"],
        "analysis_validity": analysis["analysis_validity"],
        "frame_example": {k: chosen.get(k) for k in frame_keys} if chosen else None,
        "segments_counts": {k: len(v) for k, v in analysis["segments"].items()},
        "proxy_metrics": analysis.get("proxy_metrics"),
        "task_result": analysis["task_result"],
        "feedback_policy": analysis["feedback_policy"],
    }


def write_report(path: Path, analyses: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# H2 UI-Ready Analysis Report",
        "",
        "Minimum normalized UI-ready analysis JSON export.",
        "",
        "No frontend, model architecture, training, user-facing scoring, or existing API behavior was changed.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        "- Per-sample JSON files are saved as `reports/ui_ready_analysis/<sample>/<sample>_ui_ready_analysis.json`.",
        "",
        "## Summary",
        "",
        "| Sample | Task type | Input type | Frames | Voiced % | F0 frames | Notes | Phrases | Dropouts | Breath proxy regions | Tone proxy regions | Task status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    summaries = [summarize_analysis(item) for item in analyses]
    for item in summaries:
        lines.append(
            "| {sample} | `{task}` | `{input_type}` | {frames} | {voiced:.1f}% | {f0_count} | {notes} | {phrases} | {dropouts} | {breath} | {tone} | `{status}` |".format(
                sample=f"`{item['sample']}`",
                task=item["task_type"],
                input_type=item["input_type"],
                frames=item["frames"],
                voiced=item["selected_voiced_percentage"],
                f0_count=item["selected_f0_frame_count"],
                notes=item["notes"],
                phrases=item["phrases"],
                dropouts=item["dropouts"],
                breath=item["breath_phrase_proxy_regions"],
                tone=item["tone_consistency_proxy_regions"],
                status=item["task_status"],
            )
        )

    lines += [
        "",
        "## JSON Examples",
        "",
        "Examples are compacted. Full frame arrays and debug `source_values` are in the JSON artifacts.",
        "",
    ]
    labels = {
        "00_silence": "invalid/no_voice",
        "01_speaking_voice": "speech_like",
        "03_sustained_aaa": "sustained_note",
        "04_pitch_slide": "pitch_slide",
        "05_twinkle_twinkle": "free_singing",
    }
    for analysis in analyses:
        sample = Path(analysis["input_path"]).stem
        lines += [
            f"### {labels.get(sample, sample)}: `{sample}`",
            "",
            "```json",
            json.dumps(compact_example(analysis), indent=2),
            "```",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- `frames` include selected f0/VAD source fields and debug `source_values` by default.",
        "- `segments` are frame-derived starter regions, not final production segmentation.",
        "- `task_result` and `feedback_policy` reuse existing evaluation semantics to avoid changing user-facing behavior.",
        "- H4 proxy fields are labeled `is_proxy: true` and are not breath-support, timbre, strain, or technique diagnoses.",
        "- `target_f0_hz`, `cents_error`, and full reference-song scoring remain future H3/H6 work.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ui_ready_analysis"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=h1b.h1.NANOPITCH_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = args.files if args.files else SELF_RECORDED
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyses = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Exporting H2 UI-ready analysis for {path}")
        analysis = build_ui_ready_analysis(
            path,
            checkpoint=args.checkpoint,
            nanopitch_checkpoint=args.nanopitch_checkpoint,
            device=args.device,
            debug=not args.no_debug,
        )
        sample = path.stem
        sample_dir = args.output_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"{sample}_ui_ready_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        analyses.append(analysis)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps([summarize_analysis(item) for item in analyses], indent=2), encoding="utf-8")
    write_report(Path("H2_UI_READY_ANALYSIS_REPORT.md"), analyses, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary_path),
                "report": "H2_UI_READY_ANALYSIS_REPORT.md",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
