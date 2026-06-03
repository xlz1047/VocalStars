#!/usr/bin/env python3
"""First task-specific evaluators for H3 UI-ready analysis exports."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow importing from the ml_new package even when called from the scripts dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_new.inference.algorithms import hz_to_note_name as _hz_to_note_name_impl


NOTE_OFFSETS = {
    "C": -9,
    "C#": -8,
    "DB": -8,
    "D": -7,
    "D#": -6,
    "EB": -6,
    "E": -5,
    "F": -4,
    "F#": -3,
    "GB": -3,
    "G": -2,
    "G#": -1,
    "AB": -1,
    "A": 0,
    "A#": 1,
    "BB": 1,
    "B": 2,
}

BREATH_PROXY_CAVEAT = "Breath/phrase metrics are proxy features and do not diagnose breath support."
TONE_PROXY_CAVEAT = "Tone/timbre metrics are proxy features and do not diagnose timbre or technique."


def evaluate_task(analysis: dict[str, Any], task_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = task_config or analysis.get("task_config") or {}
    annotate_reference_targets(analysis, cfg)
    task_type = str(cfg.get("task_type") or "free_singing")
    if task_type == "sustained_note":
        return evaluate_sustained_note(analysis, cfg)
    if task_type == "pitch_slide":
        return evaluate_pitch_slide(analysis, cfg)
    if task_type == "free_singing":
        return evaluate_free_singing(analysis, cfg)
    if task_type == "note_match":
        return evaluate_note_match(analysis, cfg)
    if task_type == "reference_song":
        return evaluate_reference_song(analysis, cfg)
    if task_type == "scale":
        return evaluate_reference_sequence(analysis, cfg, "scale")
    if task_type == "interval":
        return evaluate_reference_sequence(analysis, cfg, "interval")
    return unsupported_task(task_type)


def frames(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return list(analysis.get("frames") or [])


def validity(analysis: dict[str, Any]) -> dict[str, Any]:
    return analysis.get("analysis_validity") or {}


def selected_f0(frames_: list[dict[str, Any]]) -> np.ndarray:
    vals = []
    for frame in frames_:
        value = frame.get("f0_hz")
        vals.append(float(value) if value is not None and float(value) > 0 else np.nan)
    return np.asarray(vals, dtype=np.float64)


def selected_voiced(frames_: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(frame.get("voiced")) for frame in frames_], dtype=bool)


def selected_rms(frames_: list[dict[str, Any]]) -> np.ndarray:
    vals = []
    for frame in frames_:
        rms = (frame.get("volume") or {}).get("rms")
        vals.append(float(rms) if rms is not None else np.nan)
    return np.asarray(vals, dtype=np.float64)


_SINGING_F0_MIN = 60.0    # C2 — rejects FMIN boundary bins the model emits on uncertain frames
_SINGING_F0_MAX = 1500.0  # G6 — rejects extremely high spurious model outputs


def valid_f0_mask(f0: np.ndarray) -> np.ndarray:
    """Return True for frames whose F0 falls within a plausible singing range.

    The model occasionally emits its minimum or maximum pitch bin when voicing
    confidence is low. Accepting those values (32.7 Hz, 1886 Hz) as valid F0
    inflates stability and steadiness metrics to zero. Clamping to [60, 1500]
    Hz covers the full practical human vocal range without admitting boundary
    artifacts.
    """
    return np.isfinite(f0) & (f0 >= _SINGING_F0_MIN) & (f0 <= _SINGING_F0_MAX)


def cents(values_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    vals = np.asarray(values_hz, dtype=np.float64)
    out = np.full_like(vals, np.nan, dtype=np.float64)
    mask = valid_f0_mask(vals) & (reference_hz > 0)
    out[mask] = 1200.0 * np.log2(vals[mask] / reference_hz)
    return out


def f0_stability_cents(f0: np.ndarray) -> float | None:
    mask = valid_f0_mask(f0)
    vals = f0[mask]
    if vals.size < 3:
        return None
    med = float(np.median(vals))
    offsets = cents(vals, med)
    offsets = offsets[np.isfinite(offsets)]
    if offsets.size < 3:
        return None
    lo, hi = np.percentile(offsets, [5, 95])
    trimmed = offsets[(offsets >= lo) & (offsets <= hi)]
    if trimmed.size < 3:
        trimmed = offsets
    return float(np.std(trimmed))


def reference_sequence(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    reference = cfg.get("reference") or {}
    target = cfg.get("target") or {}

    # ── Dense contour format (human reference catalog API) ───────────────────
    # Detected by the presence of a positive scalar hop_s alongside an f0_hz
    # array. Each element is one time frame; duration_s = hop_s per frame.
    # Includes unvoiced frames (f0=0) so that timing is preserved correctly for
    # the voiced-span linear alignment that follows.
    ref_hop_raw = reference.get("hop_s") or target.get("hop_s")
    if ref_hop_raw is not None:
        try:
            ref_hop_s = float(ref_hop_raw)
        except (TypeError, ValueError):
            ref_hop_s = 0.0
        if ref_hop_s > 0:
            f0s_raw = reference.get("f0_hz") or target.get("f0_hz") or []
            voiced_raw = reference.get("voiced") or target.get("voiced") or []
            if isinstance(f0s_raw, list) and f0s_raw:
                sequence = []
                for i, raw in enumerate(f0s_raw):
                    try:
                        f0 = float(raw)
                    except (TypeError, ValueError):
                        f0 = 0.0
                    if not math.isfinite(f0):
                        f0 = 0.0
                    # Respect an explicit voiced mask when provided.
                    if voiced_raw and i < len(voiced_raw):
                        is_voiced = bool(voiced_raw[i]) and f0 > 0
                    else:
                        is_voiced = f0 > 0
                    sequence.append(
                        {
                            "index": i,
                            "note": _hz_to_note_name_impl(f0) if is_voiced else None,
                            "f0_hz": f0 if is_voiced else 0.0,
                            "start_s": i * ref_hop_s,
                            "end_s": (i + 1) * ref_hop_s,
                            "duration_s": ref_hop_s,
                        }
                    )
                return sequence

    # ── Legacy note-sequence format (MIDI / synth references) ────────────────
    f0s_raw = reference.get("f0_hz") or target.get("f0_hz") or []
    notes_raw = reference.get("notes") or target.get("notes") or []
    durations_raw = reference.get("durations_s") or target.get("durations_s") or []

    f0s = []
    if isinstance(f0s_raw, list):
        for value in f0s_raw:
            try:
                f0 = float(value)
            except (TypeError, ValueError):
                continue
            if f0 > 0 and math.isfinite(f0):
                f0s.append(f0)
    notes = [str(item) for item in notes_raw] if isinstance(notes_raw, list) else []
    durations = []
    if isinstance(durations_raw, list):
        for value in durations_raw:
            try:
                duration = float(value)
            except (TypeError, ValueError):
                duration = 0.0
            durations.append(duration if duration > 0 and math.isfinite(duration) else 0.7)

    sequence = []
    cursor = 0.0
    for idx, f0 in enumerate(f0s):
        duration = durations[idx] if idx < len(durations) else 0.7
        sequence.append(
            {
                "index": idx,
                "note": notes[idx] if idx < len(notes) else None,
                "f0_hz": f0,
                "start_s": cursor,
                "end_s": cursor + duration,
                "duration_s": duration,
            }
        )
        cursor += duration
    return sequence


def median_frame_hop_s(frames_: list[dict[str, Any]]) -> float:
    if len(frames_) < 2:
        return 0.01
    times = []
    for frame in frames_:
        try:
            times.append(float(frame.get("time_s") or 0.0))
        except (TypeError, ValueError):
            continue
    if len(times) < 2:
        return 0.01
    diffs = np.diff(np.asarray(times, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.01
    return float(np.median(diffs))


def aligned_reference_sequence(
    sequence: list[dict[str, Any]],
    frames_: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Linearly align a note sequence to the detected voiced/f0 span.

    This is deliberately lightweight. It makes beginner reference practice
    tolerant to starting a little late or singing a phrase faster/slower, but it
    is not beat tracking or full reference-song alignment.
    """
    if not sequence:
        return sequence, {"method": "none", "status": "missing_reference"}

    reference_duration = max(float(sequence[-1]["end_s"]), 1e-9)
    if not frames_:
        return sequence, {
            "method": "none",
            "status": "missing_frames",
            "reference_duration_s": round(reference_duration, 6),
        }

    f0 = selected_f0(frames_)
    voiced = selected_voiced(frames_)
    active = valid_f0_mask(f0) | voiced
    times = np.asarray([float(frame.get("time_s") or 0.0) for frame in frames_], dtype=np.float64)
    hop_s = median_frame_hop_s(frames_)
    full_start = float(times[0]) if times.size else 0.0
    full_end = float(times[-1] + hop_s) if times.size else reference_duration

    if np.count_nonzero(active) >= 3:
        active_indices = np.where(active)[0]
        start_s = float(times[int(active_indices[0])])
        end_s = float(times[int(active_indices[-1])] + hop_s)
        status = "aligned_to_voiced_span"
    else:
        start_s = full_start
        end_s = full_end
        status = "aligned_to_audio_span"

    aligned_duration = max(end_s - start_s, hop_s)
    tempo_scale = aligned_duration / reference_duration
    aligned = []
    for item in sequence:
        item_start = start_s + float(item["start_s"]) * tempo_scale
        item_duration = max(float(item["duration_s"]) * tempo_scale, hop_s)
        aligned.append(
            {
                **item,
                "reference_start_s": float(item["start_s"]),
                "reference_end_s": float(item["end_s"]),
                "reference_duration_s": float(item["duration_s"]),
                "start_s": item_start,
                "end_s": item_start + item_duration,
                "duration_s": item_duration,
            }
        )

    return aligned, {
        "method": "voiced_span_linear_time_warp",
        "status": status,
        "reference_duration_s": round(reference_duration, 6),
        "aligned_start_s": round(start_s, 6),
        "aligned_end_s": round(end_s, 6),
        "aligned_duration_s": round(aligned_duration, 6),
        "tempo_scale": round(tempo_scale, 6),
        "active_frame_count": int(np.count_nonzero(active)),
        "frame_count": len(frames_),
        "caveat": "Lightweight voiced-span alignment only; not beat tracking or full reference-song timing alignment.",
    }


def target_for_time(sequence: list[dict[str, Any]], time_s: float) -> dict[str, Any] | None:
    if not sequence:
        return None
    for item in sequence:
        if float(item["start_s"]) <= time_s < float(item["end_s"]):
            return item
    if abs(time_s - float(sequence[-1]["end_s"])) < 1e-6:
        return sequence[-1]
    return None


def _annotate_pitch_slide_targets(analysis: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Annotate frames with a linearly-interpolated target for pitch-slide tasks.

    Returns True when annotation was applied so the caller knows to skip
    the generic reference-sequence path.
    """
    target_cfg = cfg.get("target") or {}
    ref_cfg = cfg.get("reference") or {}
    start_hz: float | None = None
    end_hz: float | None = None
    for src in (target_cfg, ref_cfg):
        if start_hz is None:
            try:
                v = src.get("start_f0_hz")
                if v is not None:
                    start_hz = float(v)
            except (TypeError, ValueError):
                pass
        if end_hz is None:
            try:
                v = src.get("end_f0_hz")
                if v is not None:
                    end_hz = float(v)
            except (TypeError, ValueError):
                pass

    if not (start_hz and end_hz and start_hz > 0 and end_hz > 0):
        return False

    fs = analysis.get("frames") or []
    if not fs:
        return False

    n = len(fs)
    for idx, frame in enumerate(fs):
        frac = idx / max(n - 1, 1)
        target_hz = start_hz + (end_hz - start_hz) * frac
        frame["target_f0_hz"] = round(target_hz, 6)
        frame["target_note"] = _hz_to_note_name(target_hz)
        f0 = frame.get("f0_hz")
        try:
            sung = float(f0) if f0 is not None else 0.0
        except (TypeError, ValueError):
            sung = 0.0
        frame["cents_error"] = round(1200.0 * math.log2(sung / target_hz), 6) if sung > 0 else None

    return True


def annotate_reference_targets(analysis: dict[str, Any], cfg: dict[str, Any]) -> None:
    # Pitch-slide tasks prefer linear interpolation between start/end Hz.
    # Only return early if annotation actually ran; otherwise fall through to
    # the generic sequence path (which will also exit cleanly on empty sequence).
    task_type = str((cfg.get("task_type") or "")).lower()
    if task_type == "pitch_slide" and _annotate_pitch_slide_targets(analysis, cfg):
        return

    sequence = reference_sequence(cfg)
    if not sequence:
        return
    sequence, alignment = aligned_reference_sequence(sequence, frames(analysis))
    analysis["reference_alignment"] = alignment
    for frame in analysis.get("frames") or []:
        target = target_for_time(sequence, float(frame.get("time_s") or 0.0))
        if not target:
            frame.setdefault("target_f0_hz", None)
            frame.setdefault("target_note", None)
            frame.setdefault("cents_error", None)
            continue
        target_hz = float(target["f0_hz"])
        frame["target_f0_hz"] = round(target_hz, 6)
        # Prefer the note name from the reference sequence; fall back to computing it.
        note_from_seq = target.get("note")
        frame["target_note"] = note_from_seq if note_from_seq else _hz_to_note_name(target_hz)
        f0 = frame.get("f0_hz")
        if f0 is None:
            frame["cents_error"] = None
            continue
        try:
            sung = float(f0)
        except (TypeError, ValueError):
            frame["cents_error"] = None
            continue
        frame["cents_error"] = round(1200.0 * math.log2(sung / target_hz), 6) if sung > 0 else None


def reference_target_arrays(frames_: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    target = []
    errors = []
    for frame in frames_:
        target_hz = frame.get("target_f0_hz")
        error = frame.get("cents_error")
        target.append(float(target_hz) if target_hz is not None else np.nan)
        errors.append(float(error) if error is not None else np.nan)
    return np.asarray(target, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def per_note_reference_results(frames_: list[dict[str, Any]], sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    f0 = selected_f0(frames_)
    results = []
    for item in sequence:
        start = float(item["start_s"])
        end = float(item["end_s"])
        idx = [
            i
            for i, frame in enumerate(frames_)
            if start <= float(frame.get("time_s") or 0.0) < end
        ]
        if not idx:
            median_f0 = None
            median_error = None
            coverage = 0.0
        else:
            vals = f0[np.asarray(idx, dtype=int)]
            mask = valid_f0_mask(vals)
            coverage = float(np.mean(mask)) if mask.size else 0.0
            if np.any(mask):
                median_f0 = float(np.median(vals[mask]))
                median_error = 1200.0 * math.log2(median_f0 / float(item["f0_hz"]))
            else:
                median_f0 = None
                median_error = None
        results.append(
            {
                "index": item["index"],
                "note": item.get("note"),
                "target_f0_hz": round(float(item["f0_hz"]), 6),
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "duration_s": round(float(item["duration_s"]), 6),
                "sung_median_f0_hz": round(median_f0, 6) if median_f0 is not None else None,
                "median_cents_error": round(median_error, 6) if median_error is not None else None,
                "f0_coverage": round(coverage, 6),
            }
        )
    return results


def contour_direction_agreement(note_results: list[dict[str, Any]]) -> float | None:
    comparisons = []
    for prev, cur in zip(note_results[:-1], note_results[1:]):
        prev_target = prev.get("target_f0_hz")
        cur_target = cur.get("target_f0_hz")
        prev_sung = prev.get("sung_median_f0_hz")
        cur_sung = cur.get("sung_median_f0_hz")
        if prev_target is None or cur_target is None or prev_sung is None or cur_sung is None:
            continue
        target_delta = float(cur_target) - float(prev_target)
        sung_delta = float(cur_sung) - float(prev_sung)
        if abs(target_delta) < 1e-6:
            comparisons.append(abs(sung_delta) < 10.0)
        else:
            comparisons.append((target_delta > 0 and sung_delta > 0) or (target_delta < 0 and sung_delta < 0))
    if not comparisons:
        return None
    return float(np.mean(comparisons))


def reference_pitch_error_regions(note_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions = []
    for item in note_results:
        error = item.get("median_cents_error")
        coverage = float(item.get("f0_coverage") or 0.0)
        has_pitch_error = error is not None and abs(float(error)) > 75.0
        has_low_coverage = coverage < 0.35
        if not has_pitch_error and not has_low_coverage:
            continue

        severity = "warning"
        reasons = []
        if error is not None:
            abs_error = abs(float(error))
            if abs_error > 150.0:
                severity = "error"
            if abs_error > 75.0:
                direction = "sharp" if float(error) > 0 else "flat"
                reasons.append(f"{round(abs_error)} cents {direction}")
        if has_low_coverage:
            severity = "error" if coverage < 0.15 else severity
            reasons.append("low f0 coverage")

        note_label = item.get("note") or f"note {int(item.get('index') or 0) + 1}"
        regions.append(
            {
                "id": f"reference_pitch_error_{int(item.get('index') or 0) + 1:03d}",
                "type": "reference_pitch_error",
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
                "duration_s": item.get("duration_s"),
                "target_f0_hz": item.get("target_f0_hz"),
                "sung_median_f0_hz": item.get("sung_median_f0_hz"),
                "median_cents_error": item.get("median_cents_error"),
                "f0_coverage": item.get("f0_coverage"),
                "source": "reference_contour_alignment",
                "ui_severity": severity,
                "summary": f"{note_label}: {', '.join(reasons)}",
                "actionable_hint": "Replay this note region, listen to the target, then sing the note slowly before reconnecting the phrase.",
                "caveats": ["Reference note markers use provisional f0-contour alignment, not full rhythm scoring."],
            }
        )
    return regions


def f0_drift_cents(f0: np.ndarray) -> float | None:
    mask = valid_f0_mask(f0)
    idx = np.where(mask)[0]
    if idx.size < 6:
        return None
    chunk = max(3, idx.size // 10)
    start_vals = f0[idx[:chunk]]
    end_vals = f0[idx[-chunk:]]
    start = float(np.median(start_vals))
    end = float(np.median(end_vals))
    if start <= 0 or end <= 0:
        return None
    return float(1200.0 * math.log2(end / start))


def f0_range_cents(f0: np.ndarray) -> float | None:
    vals = f0[valid_f0_mask(f0)]
    if vals.size < 3:
        return None
    p05, p95 = np.percentile(vals, [5, 95])
    if p05 <= 0:
        return None
    return float(1200.0 * math.log2(float(p95) / float(p05)))


def direction_slope_hz_per_s(f0: np.ndarray, hop_s: float) -> float | None:
    idx = np.where(valid_f0_mask(f0))[0]
    if idx.size < 3:
        return None
    x = idx.astype(np.float64) * hop_s
    y = f0[idx]
    return float(np.polyfit(x, y, 1)[0])


def dropout_rate_from_f0(f0: np.ndarray) -> float:
    mask = valid_f0_mask(f0)
    if not np.any(mask):
        return 1.0
    first = int(np.argmax(mask))
    last = len(mask) - 1 - int(np.argmax(mask[::-1]))
    active = mask[first : last + 1]
    return float(np.mean(~active)) if active.size else 1.0


def voiced_continuity(voiced: np.ndarray) -> float:
    if voiced.size == 0:
        return 0.0
    return float(np.mean(voiced))


def volume_steadiness_score(rms: np.ndarray, active_mask: np.ndarray) -> tuple[float | None, float | None]:
    mask = active_mask & np.isfinite(rms) & (rms > 0)
    vals = rms[mask]
    if vals.size < 5:
        return None, None
    db = 20.0 * np.log10(np.maximum(vals, 1e-12))
    std = float(np.std(db))
    score = clamp01(1.0 - std / 12.0)
    return score, std


def signal_quality_score(frames_: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    if not frames_:
        return 0.0, {}
    clipped = np.mean([bool((f.get("signal_quality") or {}).get("clipped")) for f in frames_])
    near_silence = np.mean([bool((f.get("signal_quality") or {}).get("near_silence")) for f in frames_])
    noisy = np.mean([bool((f.get("signal_quality") or {}).get("noisy")) for f in frames_])
    low_conf = np.mean([bool((f.get("signal_quality") or {}).get("low_confidence")) for f in frames_])
    disagreement = np.mean([bool((f.get("signal_quality") or {}).get("source_disagreement")) for f in frames_])
    penalty = clipped * 0.4 + near_silence * 0.2 + noisy * 0.25 + low_conf * 0.25 + disagreement * 0.25
    return clamp01(1.0 - penalty), {
        "clipped_frame_ratio": float(clipped),
        "near_silence_frame_ratio": float(near_silence),
        "noisy_frame_ratio": float(noisy),
        "low_confidence_frame_ratio": float(low_conf),
        "source_disagreement_frame_ratio": float(disagreement),
    }


def proxy_subscores(analysis: dict[str, Any]) -> dict[str, Any]:
    proxies = analysis.get("proxy_metrics") or {}
    if not proxies:
        return {}
    return {
        "breath_phrase_proxy": proxies.get("breath_phrase"),
        "tone_consistency_proxy": proxies.get("tone_consistency"),
        "voice_quality_proxy": proxies.get("voice_quality_proxy"),
    }


def phrase_continuity_score(segments: dict[str, Any], total_frames: int) -> float:
    phrases = segments.get("phrases") or []
    if total_frames <= 0:
        return 0.0
    if not phrases:
        return 0.0
    total_phrase_s = sum(float(p.get("duration_s") or 0.0) for p in phrases)
    coverage = total_phrase_s / max(total_frames * 0.01, 1e-9)
    fragmentation_penalty = max(0, len(phrases) - 1) * 0.04
    return clamp01(coverage - fragmentation_penalty)


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def score_from_components(*components: float | None) -> int | None:
    vals = [clamp01(c) for c in components if c is not None]
    if not vals:
        return None
    return int(round(100.0 * float(np.mean(vals))))


def invalid_if_needed(analysis: dict[str, Any], task_type: str) -> dict[str, Any] | None:
    v = validity(analysis)
    input_type = v.get("input_type")
    if input_type in {"no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"}:
        summary = {
            "no_voice_or_noise": "No analyzable singing was detected.",
            "speech_like_or_non_singing": "This sounds like speech or non-singing voice, so task coaching was not generated.",
            "low_confidence_or_unreliable": "The audio was too noisy or unreliable to score confidently.",
        }.get(input_type, "Input was not suitable for task scoring.")
        return result(
            task_type=task_type,
            status=f"{input_type}_no_task_score",
            full_song_score=None,
            diagnostic_score=None,
            summary=summary,
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[
                blocked("task_score", "Input did not pass analysis validity checks."),
                blocked("singing_exercises", "Input did not pass analysis validity checks."),
                blocked("note_specific_pitch_advice", "Input did not pass analysis validity checks."),
            ],
            caveats=["Task evaluator skipped because input was not analyzable for this task."],
            next_exercise_suggestion=None,
        )
    return None


def evaluate_sustained_note(analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    skipped = invalid_if_needed(analysis, "sustained_note")
    if skipped:
        return skipped
    fs = frames(analysis)
    f0 = selected_f0(fs)
    voiced = selected_voiced(fs)
    rms = selected_rms(fs)
    active = valid_f0_mask(f0)
    f0_coverage = float(np.mean(active)) if active.size else 0.0
    stability = f0_stability_cents(f0)
    drift = f0_drift_cents(f0)
    dropout = dropout_rate_from_f0(f0)
    continuity = max(voiced_continuity(voiced), f0_coverage)
    volume_score, volume_std_db = volume_steadiness_score(rms, active)
    stability_score = clamp01(1.0 - (stability or 999.0) / 120.0)
    drift_score = clamp01(1.0 - abs(drift or 999.0) / 150.0)
    dropout_score = clamp01(1.0 - dropout)
    continuity_score = clamp01(continuity)
    diagnostic = score_from_components(stability_score, drift_score, continuity_score, dropout_score, volume_score)
    subscores = {
        "pitch_stability": round(stability_score, 6),
        "pitch_drift": round(drift_score, 6),
        "voiced_continuity": round(continuity_score, 6),
        "dropout_rate": round(dropout, 6),
        "volume_steadiness": round(volume_score, 6) if volume_score is not None else None,
        "raw_metrics": {
            "f0_stability_cents": round(stability, 6) if stability is not None else None,
            "f0_drift_cents": round(drift, 6) if drift is not None else None,
            "volume_std_db": round(volume_std_db, 6) if volume_std_db is not None else None,
            "selected_f0_coverage": round(f0_coverage, 6),
            "selected_voiced_coverage": round(voiced_continuity(voiced), 6),
        },
        "proxy_metrics": proxy_subscores(analysis),
    }
    next_ex = None
    if stability is not None and stability > 80:
        next_ex = exercise("sustain_steady_pitch", "Hold one comfortable note for 4 seconds", "Pitch stability varied during the sustained tone.")
    elif dropout > 0.15:
        next_ex = exercise("sustain_connected_tone", "Sustain through the full vowel", "There were gaps in the selected f0 contour.")
    return result(
        task_type="sustained_note",
        status="diagnostic_sustained_note_complete",
        full_song_score=None,
        diagnostic_score=diagnostic,
        summary="Sustained-note diagnostic computed from selected f0, voicing, and volume continuity.",
        subscores=subscores,
        allowed_feedback=["pitch_stability", "pitch_drift", "voicing_continuity", "signal_quality", "breath_phrase_proxy", "tone_consistency_proxy"],
        blocked_feedback=[
            blocked("full_song_score", "Sustained-note tasks are diagnostic-only."),
            blocked("reference_melody_accuracy", "No reference melody was evaluated."),
            blocked("timbre_diagnosis", "Timbre labels are not reliable enough for coaching."),
        ],
        caveats=["Diagnostic sustained-note score only; no reference melody was evaluated.", BREATH_PROXY_CAVEAT, TONE_PROXY_CAVEAT],
        next_exercise_suggestion=next_ex,
    )


def _slide_endpoint_accuracy(f0: np.ndarray, target_hz: float, window_frames: int = 15) -> tuple[float, float | None]:
    """Return (score 0-1, median_hz | None) for the start or end window of a slide."""
    idx = np.where(valid_f0_mask(f0))[0]
    if idx.size < 3 or target_hz <= 0:
        return 0.5, None  # not enough data → neutral
    vals = f0[idx[:window_frames]]
    if vals.size == 0:
        return 0.5, None
    median_hz = float(np.median(vals))
    abs_cents = abs(1200.0 * math.log2(median_hz / target_hz))
    # Score: full marks within 50 cents, zero at 600 cents (5 semitones)
    score = clamp01(1.0 - abs_cents / 600.0)
    return score, median_hz


def _slide_contour_score(f0: np.ndarray, start_hz: float, end_hz: float) -> tuple[float, float | None]:
    """Compare the detected F0 against an ideal linear slide from start_hz to end_hz."""
    idx = np.where(valid_f0_mask(f0))[0]
    if idx.size < 4 or start_hz <= 0 or end_hz <= 0:
        return 0.5, None
    n = len(f0)
    ideal = np.array([start_hz + (end_hz - start_hz) * (i / max(n - 1, 1)) for i in range(n)])
    errors_cents = []
    for i in idx:
        if ideal[i] > 0:
            errors_cents.append(abs(1200.0 * math.log2(float(f0[i]) / ideal[i])))
    if not errors_cents:
        return 0.5, None
    rms_cents = float(np.sqrt(np.mean(np.square(errors_cents))))
    # A perfect slide → 0 cents RMS; 400 cents RMS → score 0
    score = clamp01(1.0 - rms_cents / 400.0)
    return score, rms_cents


def _slide_feedback(
    direction_correct: bool,
    expected: str,
    detected: str,
    start_score: float | None,
    end_score: float | None,
    start_target_hz: float | None,
    end_target_hz: float | None,
    start_sung_hz: float | None,
    end_sung_hz: float | None,
    smoothness_score: float,
    contour_score: float | None,
) -> dict[str, str]:
    fb: dict[str, str] = {}
    if not direction_correct:
        fb["direction"] = (
            f"You slid {detected}, but the exercise calls for a {expected} slide. "
            "Try starting on your chosen note and gliding clearly in the other direction."
        )
    else:
        fb["direction"] = f"Direction correct — you slid {detected}."

    if start_score is not None and start_target_hz is not None:
        if start_score < 0.5 and start_sung_hz is not None:
            diff_st = round(1200.0 * math.log2(start_sung_hz / start_target_hz))
            note_target = _hz_to_note_name(start_target_hz)
            fb["start_note"] = (
                f"Start note: you began around {start_sung_hz:.0f} Hz, "
                f"but the target was {note_target} ({start_target_hz:.0f} Hz), "
                f"a difference of {diff_st:+d} cents."
            )
        elif start_score >= 0.8:
            fb["start_note"] = "Start note: well matched."

    if end_score is not None and end_target_hz is not None:
        if end_score < 0.5 and end_sung_hz is not None:
            diff_en = round(1200.0 * math.log2(end_sung_hz / end_target_hz))
            note_target = _hz_to_note_name(end_target_hz)
            fb["end_note"] = (
                f"End note: you finished around {end_sung_hz:.0f} Hz, "
                f"but the target was {note_target} ({end_target_hz:.0f} Hz), "
                f"a difference of {diff_en:+d} cents."
            )
        elif end_score >= 0.8:
            fb["end_note"] = "End note: well matched."

    if smoothness_score < 0.5:
        fb["smoothness"] = "Smoothness: several jumpy transitions detected. Aim for a steady, continuous glide."
    elif smoothness_score >= 0.8:
        fb["smoothness"] = "Smoothness: the glide was fluid."

    if contour_score is not None and contour_score < 0.5:
        fb["contour"] = "Contour tracking: the slide shape diverged noticeably from the ideal linear path."
    elif contour_score is not None and contour_score >= 0.8:
        fb["contour"] = "Contour tracking: closely followed the target path."

    return fb


def _hz_to_note_name(hz: float) -> str:
    """Wrapper for the canonical implementation in ml_new.inference.algorithms."""
    if hz <= 0 or not math.isfinite(hz):
        return "--"
    return _hz_to_note_name_impl(float(hz))


def evaluate_pitch_slide(analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    skipped = invalid_if_needed(analysis, "pitch_slide")
    if skipped:
        return skipped

    fs = frames(analysis)
    f0 = selected_f0(fs)
    voiced = selected_voiced(fs)
    active = valid_f0_mask(f0)
    f0_coverage = float(np.mean(active)) if active.size else 0.0
    hop_s = 0.01

    # ── Direction sub-score ──────────────────────────────────────────────────
    slope = direction_slope_hz_per_s(f0, hop_s)
    detected_direction = "flat"
    if slope is not None and slope > 2.0:
        detected_direction = "up"
    elif slope is not None and slope < -2.0:
        detected_direction = "down"

    expected = str(
        cfg.get("expected_direction")
        or (cfg.get("target") or {}).get("direction")
        or "any"
    ).lower()

    if expected in {"up_or_down", "any"}:
        direction_correct = detected_direction in {"up", "down"}
        direction_score = 1.0 if direction_correct else 0.2
    elif expected == detected_direction:
        direction_correct = True
        direction_score = 1.0
    else:
        direction_correct = False
        direction_score = 0.0  # wrong direction → zero, triggers cap

    # ── Endpoint accuracy (when target start/end Hz are provided) ────────────
    target_cfg = cfg.get("target") or {}
    ref_cfg = cfg.get("reference") or {}
    start_target_hz: float | None = None
    end_target_hz: float | None = None
    for src in (target_cfg, ref_cfg):
        if start_target_hz is None and src.get("start_f0_hz"):
            try:
                start_target_hz = float(src["start_f0_hz"])
            except (TypeError, ValueError):
                pass
        if end_target_hz is None and src.get("end_f0_hz"):
            try:
                end_target_hz = float(src["end_f0_hz"])
            except (TypeError, ValueError):
                pass

    start_score: float | None = None
    start_sung_hz: float | None = None
    end_score: float | None = None
    end_sung_hz: float | None = None

    if start_target_hz and start_target_hz > 0:
        start_score, start_sung_hz = _slide_endpoint_accuracy(f0, start_target_hz, window_frames=15)
    if end_target_hz and end_target_hz > 0:
        # End window: use last 15 voiced frames
        f0_rev = f0[::-1]
        end_score_val, end_sung_rev = _slide_endpoint_accuracy(f0_rev, end_target_hz, window_frames=15)
        end_score = end_score_val
        end_sung_hz = end_sung_rev

    # ── Smoothness sub-score ─────────────────────────────────────────────────
    range_cents = f0_range_cents(f0)
    range_score = clamp01((range_cents or 0.0) / 700.0)
    deltas = successive_cents_deltas(f0)
    smoothness_raw = float(np.percentile(np.abs(deltas), 90)) if deltas.size else None
    smoothness_score = clamp01(1.0 - (smoothness_raw or 999.0) / 250.0)

    # ── Contour tracking (only when both endpoints known) ────────────────────
    contour_score: float | None = None
    contour_rms: float | None = None
    if start_target_hz and end_target_hz and start_target_hz > 0 and end_target_hz > 0:
        contour_score, contour_rms = _slide_contour_score(f0, start_target_hz, end_target_hz)

    # ── Continuity ───────────────────────────────────────────────────────────
    dropout = dropout_rate_from_f0(f0)
    continuity = max(voiced_continuity(voiced), f0_coverage)

    # ── Composite score ──────────────────────────────────────────────────────
    components = [direction_score, range_score, smoothness_score, 1.0 - dropout, continuity]
    if start_score is not None:
        components.append(start_score)
    if end_score is not None:
        components.append(end_score)
    if contour_score is not None:
        components.append(contour_score)

    raw_score = score_from_components(*components)

    # Cap at 70 when direction is wrong or start/end note badly missed
    score_capped = False
    if not direction_correct:
        score_capped = True
    elif start_score is not None and start_score < 0.15:
        score_capped = True
    elif end_score is not None and end_score < 0.15:
        score_capped = True

    diagnostic = min(raw_score or 0, 70) if score_capped else raw_score

    # ── Feedback generation ──────────────────────────────────────────────────
    feedback = _slide_feedback(
        direction_correct=direction_correct,
        expected=expected,
        detected=detected_direction,
        start_score=start_score,
        end_score=end_score,
        start_target_hz=start_target_hz,
        end_target_hz=end_target_hz,
        start_sung_hz=start_sung_hz,
        end_sung_hz=end_sung_hz,
        smoothness_score=smoothness_score,
        contour_score=contour_score,
    )

    subscores: dict[str, Any] = {
        "direction": round(direction_score, 6),
        "direction_correct": direction_correct,
        "range": round(range_score, 6),
        "smoothness": round(smoothness_score, 6),
        "continuity": round(continuity, 6),
        "dropout_rate": round(dropout, 6),
        "score_capped": score_capped,
        "pitch_slide_breakdown": {
            "start_note_accuracy": round(start_score, 6) if start_score is not None else None,
            "end_note_accuracy": round(end_score, 6) if end_score is not None else None,
            "direction_correct": direction_correct,
            "smoothness_score": round(smoothness_score, 6),
            "contour_deviation_score": round(contour_score, 6) if contour_score is not None else None,
            "contour_deviation_cents": round(contour_rms, 6) if contour_rms is not None else None,
            "overall": diagnostic if diagnostic is not None else 0,
            "score_capped": score_capped,
            "feedback": feedback,
        },
        "raw_metrics": {
            "direction": detected_direction,
            "direction_slope_hz_per_s": round(slope, 6) if slope is not None else None,
            "range_cents": round(range_cents, 6) if range_cents is not None else None,
            "p90_abs_step_cents": round(smoothness_raw, 6) if smoothness_raw is not None else None,
            "start_target_hz": round(start_target_hz, 6) if start_target_hz else None,
            "end_target_hz": round(end_target_hz, 6) if end_target_hz else None,
            "start_sung_hz": round(start_sung_hz, 6) if start_sung_hz else None,
            "end_sung_hz": round(end_sung_hz, 6) if end_sung_hz else None,
            "contour_rms_cents": round(contour_rms, 6) if contour_rms is not None else None,
            "selected_f0_coverage": round(f0_coverage, 6),
            "selected_voiced_coverage": round(voiced_continuity(voiced), 6),
        },
        "proxy_metrics": proxy_subscores(analysis),
    }

    next_ex = None
    if not direction_correct:
        next_ex = exercise(
            "slide_clear_direction",
            "Slide slowly in one clear direction",
            f"The detected direction ({detected_direction}) did not match the task target ({expected}).",
        )
    elif smoothness_score < 0.6:
        next_ex = exercise(
            "slow_siren",
            "Practice a slow siren between two comfortable notes",
            "The selected f0 contour had abrupt movement.",
        )
    elif start_score is not None and start_score < 0.4:
        next_ex = exercise(
            "find_start_note",
            "Listen to the start note, hum it, then begin the slide there",
            "The starting note was significantly off from the target.",
        )

    caveats = [
        "Diagnostic pitch-slide score only; no reference melody was evaluated.",
        BREATH_PROXY_CAVEAT,
        TONE_PROXY_CAVEAT,
    ]
    if score_capped:
        caveats.insert(0, "Score was capped at 70 because the direction or start/end note was significantly wrong.")

    return result(
        task_type="pitch_slide",
        status="diagnostic_pitch_slide_complete",
        full_song_score=None,
        diagnostic_score=diagnostic,
        summary=(
            "Pitch-slide diagnostic: direction, range, smoothness, endpoint accuracy, and continuity."
            + (" Score capped at 70 — direction or note target significantly missed." if score_capped else "")
        ),
        subscores=subscores,
        allowed_feedback=[
            "pitch_slide_direction",
            "pitch_slide_start_note",
            "pitch_slide_end_note",
            "pitch_slide_range",
            "pitch_slide_smoothness",
            "pitch_slide_contour",
            "voicing_continuity",
            "signal_quality",
            "breath_phrase_proxy",
            "tone_consistency_proxy",
        ],
        blocked_feedback=[
            blocked("full_song_score", "Pitch-slide tasks are diagnostic-only."),
            blocked("reference_melody_accuracy", "No reference melody was evaluated."),
            blocked("note_specific_melody_advice", "A slide is a continuous contour, not a normal melody."),
        ],
        caveats=caveats,
        next_exercise_suggestion=next_ex,
    )


def evaluate_free_singing(analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    skipped = invalid_if_needed(analysis, "free_singing")
    if skipped:
        return skipped
    fs = frames(analysis)
    f0 = selected_f0(fs)
    signal_score, signal_metrics = signal_quality_score(fs)
    stability = f0_stability_cents(f0)
    stability_score = clamp01(1.0 - (stability or 999.0) / 500.0)
    phrase_score = phrase_continuity_score(analysis.get("segments") or {}, len(fs))
    diagnostic = score_from_components(stability_score, phrase_score, signal_score)
    subscores = {
        "general_pitch_stability": round(stability_score, 6),
        "phrase_continuity": round(phrase_score, 6),
        "signal_quality": round(signal_score, 6),
        "raw_metrics": {
            "f0_stability_cents": round(stability, 6) if stability is not None else None,
            **{k: round(v, 6) for k, v in signal_metrics.items()},
        },
        "proxy_metrics": proxy_subscores(analysis),
    }
    next_ex = None
    if signal_score < 0.7:
        next_ex = exercise("recording_quality_retry", "Try again with less background noise", "Signal quality reduced confidence in the analysis.")
    elif phrase_score < 0.5:
        next_ex = exercise("connect_short_phrase", "Sing one short phrase on a comfortable vowel", "Phrase continuity was fragmented.")
    return result(
        task_type="free_singing",
        status="free_singing_general_feedback",
        full_song_score=diagnostic,
        diagnostic_score=None,
        summary="Free-singing diagnostic computed without reference-melody scoring.",
        subscores=subscores,
        allowed_feedback=["general_pitch_contour", "phrase_continuity", "signal_quality", "breath_phrase_proxy", "tone_consistency_proxy"],
        blocked_feedback=[
            blocked("reference_melody_accuracy", "No reference melody was provided."),
            blocked("timbre_diagnosis", "Timbre labels are not reliable enough for coaching."),
        ],
        caveats=[
            "Score is based on detected pitch and continuity features, not a reference melody.",
            BREATH_PROXY_CAVEAT,
            TONE_PROXY_CAVEAT,
        ],
        next_exercise_suggestion=next_ex,
    )


def evaluate_note_match(analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    target_hz = target_f0_hz(cfg)
    if target_hz is None:
        return result(
            task_type="note_match",
            status="insufficient_target_info",
            full_song_score=None,
            diagnostic_score=None,
            summary="Note-match task requires target.note or target.f0_hz.",
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[
                blocked("note_match_score", "Target note or f0 is missing."),
                blocked("pitch_accuracy", "Target note or f0 is missing."),
            ],
            caveats=["Target note or target f0 is required for note-match scoring."],
            next_exercise_suggestion=None,
        )
    skipped = invalid_if_needed(analysis, "note_match")
    if skipped:
        return skipped
    fs = frames(analysis)
    f0 = selected_f0(fs)
    mask = valid_f0_mask(f0)
    offsets = cents(f0, target_hz)
    vals = offsets[mask & np.isfinite(offsets)]
    if vals.size == 0:
        median_abs = None
        within_50 = 0.0
    else:
        median_abs = float(np.median(np.abs(vals)))
        within_50 = float(np.mean(np.abs(vals) <= 50.0))
    stability = f0_stability_cents(f0)
    accuracy_score = clamp01(1.0 - (median_abs if median_abs is not None else 999.0) / 200.0)
    stability_score = clamp01(1.0 - (stability or 999.0) / 120.0)
    coverage = float(np.mean(mask)) if mask.size else 0.0
    diagnostic = score_from_components(accuracy_score, stability_score, coverage, within_50)
    subscores = {
        "pitch_accuracy": round(accuracy_score, 6),
        "pitch_stability": round(stability_score, 6),
        "voiced_f0_coverage": round(coverage, 6),
        "within_50_cents": round(within_50, 6),
        "raw_metrics": {
            "target_f0_hz": round(target_hz, 6),
            "median_abs_cents_error": round(median_abs, 6) if median_abs is not None else None,
            "f0_stability_cents": round(stability, 6) if stability is not None else None,
        },
        "proxy_metrics": proxy_subscores(analysis),
    }
    next_ex = None
    if median_abs is not None and median_abs > 80:
        next_ex = exercise("match_target_note_slowly", "Listen, hum, then sing the target note", "The selected f0 was far from the target note.")
    return result(
        task_type="note_match",
        status="note_match_diagnostic_complete",
        full_song_score=None,
        diagnostic_score=diagnostic,
        summary="Note-match diagnostic computed against the provided target note.",
        subscores=subscores,
        allowed_feedback=["pitch_accuracy", "pitch_stability", "voicing_continuity", "signal_quality", "breath_phrase_proxy", "tone_consistency_proxy"],
        blocked_feedback=[
            blocked("full_song_score", "Note-match tasks are diagnostic-only."),
            blocked("reference_song_accuracy", "No reference melody was evaluated."),
        ],
        caveats=["Diagnostic note-match score only; no reference melody was evaluated.", BREATH_PROXY_CAVEAT, TONE_PROXY_CAVEAT],
        next_exercise_suggestion=next_ex,
    )


def evaluate_reference_song(analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    sequence = reference_sequence(cfg)
    if not sequence:
        return result(
            task_type="reference_song",
            status="insufficient_reference_info",
            full_song_score=None,
            diagnostic_score=None,
            summary="Reference-song scoring requires reference melody and timing.",
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[
                blocked("reference_melody_accuracy", "Reference melody/timing is missing."),
                blocked("full_song_score", "Reference-song score cannot be computed without a reference."),
                blocked("rhythm_accuracy", "Reference beat/timing grid is missing."),
            ],
            caveats=["Reference-song scoring is a placeholder; no fake reference score was generated."],
            next_exercise_suggestion=None,
        )
    return evaluate_reference_sequence(analysis, cfg, "reference_song")


def evaluate_reference_sequence(analysis: dict[str, Any], cfg: dict[str, Any], task_type: str) -> dict[str, Any]:
    sequence = reference_sequence(cfg)
    if not sequence:
        return result(
            task_type=task_type,
            status="insufficient_reference_info",
            full_song_score=None,
            diagnostic_score=None,
            summary="Reference contour scoring requires note f0 targets and durations.",
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[
                blocked("reference_contour_score", "Reference notes/f0/durations are missing."),
                blocked("full_song_score", "Full-song scoring requires a reference contour."),
                blocked("rhythm_accuracy", "Reference timing grid is missing."),
            ],
            caveats=["No fake reference score was generated because reference information was incomplete."],
            next_exercise_suggestion=None,
        )
    skipped = invalid_if_needed(analysis, task_type)
    if skipped:
        return skipped

    fs = frames(analysis)
    if not fs:
        return result(
            task_type=task_type,
            status="no_frames_for_reference_contour",
            full_song_score=None,
            diagnostic_score=None,
            summary="Reference contour scoring requires frame-level f0 data.",
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[blocked("reference_contour_score", "No frame-level analysis was available.")],
            caveats=["No fake reference score was generated because frames were unavailable."],
            next_exercise_suggestion=None,
        )
    sequence, alignment = aligned_reference_sequence(sequence, fs)
    analysis["reference_alignment"] = alignment
    f0 = selected_f0(fs)
    voiced = selected_voiced(fs)
    target, errors = reference_target_arrays(fs)
    target_mask = np.isfinite(target) & (target > 0)
    comparable = target_mask & np.isfinite(errors) & valid_f0_mask(f0)
    if not np.any(target_mask):
        return result(
            task_type=task_type,
            status="reference_timing_outside_audio",
            full_song_score=None,
            diagnostic_score=None,
            summary="Reference notes were supplied, but no target frames overlapped this recording.",
            subscores={},
            allowed_feedback=["signal_quality"],
            blocked_feedback=[blocked("reference_contour_score", "Reference timing did not overlap the analyzed audio.")],
            caveats=["No fake reference score was generated because reference timing did not overlap the recording."],
            next_exercise_suggestion=None,
        )

    abs_errors = np.abs(errors[comparable])
    median_abs = float(np.median(abs_errors)) if abs_errors.size else None
    mean_abs = float(np.mean(abs_errors)) if abs_errors.size else None
    within_50 = float(np.mean(abs_errors <= 50.0)) if abs_errors.size else 0.0
    within_100 = float(np.mean(abs_errors <= 100.0)) if abs_errors.size else 0.0
    f0_coverage = float(np.mean(comparable[target_mask])) if np.any(target_mask) else 0.0
    voiced_coverage = float(np.mean(voiced[target_mask])) if np.any(target_mask) else 0.0
    note_results = per_note_reference_results(fs, sequence)
    error_regions = reference_pitch_error_regions(note_results)
    segments = analysis.setdefault("segments", {})
    if isinstance(segments, dict):
        segments["reference_pitch_error_regions"] = error_regions
    direction_agreement = contour_direction_agreement(note_results)
    signal_score, signal_metrics = signal_quality_score(fs)

    pitch_accuracy = clamp01(1.0 - (median_abs if median_abs is not None else 999.0) / 250.0)
    coverage_score = clamp01(max(f0_coverage, voiced_coverage))
    direction_score = direction_agreement if direction_agreement is not None else None
    diagnostic = score_from_components(pitch_accuracy, within_50, coverage_score, direction_score, signal_score)
    status_prefix = {
        "reference_song": "provisional_reference_contour",
        "scale": "provisional_scale_contour",
        "interval": "provisional_interval_contour",
    }.get(task_type, "provisional_reference_contour")
    subscores = {
        "reference_pitch_accuracy": round(pitch_accuracy, 6),
        "within_50_cents": round(within_50, 6),
        "within_100_cents": round(within_100, 6),
        "reference_f0_coverage": round(f0_coverage, 6),
        "reference_voiced_coverage": round(voiced_coverage, 6),
        "contour_direction_agreement": round(direction_agreement, 6) if direction_agreement is not None else None,
        "signal_quality": round(signal_score, 6),
        "raw_metrics": {
            "median_abs_cents_error": round(median_abs, 6) if median_abs is not None else None,
            "mean_abs_cents_error": round(mean_abs, 6) if mean_abs is not None else None,
            "reference_note_count": len(sequence),
            **{k: round(v, 6) for k, v in signal_metrics.items()},
        },
        "reference_alignment": alignment,
        "reference_pitch_error_region_count": len(error_regions),
        "note_results": note_results,
        "proxy_metrics": proxy_subscores(analysis),
    }
    next_ex = None
    if median_abs is not None and median_abs > 120:
        next_ex = exercise("slow_reference_notes", "Practice the melody one note at a time", "Several sung notes were far from the reference contour.")
    elif f0_coverage < 0.55:
        next_ex = exercise("connect_reference_phrase", "Sing through the whole short phrase", "The selected f0 contour did not cover much of the reference.")
    elif direction_agreement is not None and direction_agreement < 0.7:
        next_ex = exercise("trace_melody_direction", "Listen for whether each next note goes up, down, or repeats", "The sung contour direction often differed from the reference.")

    return result(
        task_type=task_type,
        status=f"{status_prefix}_complete",
        full_song_score=None,
        diagnostic_score=diagnostic,
        summary="Provisional reference-contour diagnostic computed from selected f0 against the supplied note sequence.",
        subscores=subscores,
        allowed_feedback=[
            "reference_pitch_contour",
            "note_match_regions",
            "voicing_continuity",
            "signal_quality",
            "breath_phrase_proxy",
            "tone_consistency_proxy",
        ],
        blocked_feedback=[
            blocked("full_song_score", "Full reference-song scoring still requires robust alignment and rhythm evaluation."),
            blocked("rhythm_accuracy", "Rhythm scoring is not implemented for this provisional contour diagnostic."),
            blocked("background_track_accuracy", "Only the sung f0 contour is evaluated, not accompaniment or production."),
        ],
        caveats=[
            "This is provisional reference-contour feedback, not full reference-song accuracy.",
            "Reference timing uses lightweight voiced-span alignment; rhythm and beat accuracy are not evaluated yet.",
            BREATH_PROXY_CAVEAT,
            TONE_PROXY_CAVEAT,
        ],
        next_exercise_suggestion=next_ex,
    )


def successive_cents_deltas(f0: np.ndarray) -> np.ndarray:
    idx = np.where(valid_f0_mask(f0))[0]
    deltas = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        deltas.append(1200.0 * math.log2(float(f0[b]) / max(float(f0[a]), 1e-9)))
    return np.asarray(deltas, dtype=np.float64)


def target_f0_hz(cfg: dict[str, Any]) -> float | None:
    target = cfg.get("target") or {}
    if target.get("f0_hz") is not None:
        try:
            value = float(target["f0_hz"])
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None
    note = target.get("note")
    if note:
        return note_to_hz(str(note))
    return None


def note_to_hz(note: str) -> float | None:
    cleaned = note.strip().upper().replace("♯", "#").replace("♭", "B")
    if not cleaned:
        return None
    if len(cleaned) >= 2 and cleaned[1] in {"#", "B"}:
        name = cleaned[:2]
        octave_text = cleaned[2:]
    else:
        name = cleaned[:1]
        octave_text = cleaned[1:]
    if name not in NOTE_OFFSETS:
        return None
    try:
        octave = int(octave_text)
    except ValueError:
        return None
    semitones_from_a4 = NOTE_OFFSETS[name] + (octave - 4) * 12
    return float(440.0 * (2.0 ** (semitones_from_a4 / 12.0)))


def blocked(kind: str, reason: str) -> dict[str, str]:
    return {"type": kind, "reason": reason}


def exercise(exercise_id: str, title: str, reason: str) -> dict[str, str]:
    return {"exercise_id": exercise_id, "title": title, "reason": reason}


def result(
    *,
    task_type: str,
    status: str,
    full_song_score: int | None,
    diagnostic_score: int | None,
    summary: str,
    subscores: dict[str, Any],
    allowed_feedback: list[str],
    blocked_feedback: list[dict[str, str]],
    caveats: list[str],
    next_exercise_suggestion: dict[str, str] | None,
) -> dict[str, Any]:
    proxy_blockers = [
        blocked("breath_support_claims", "Breath/phrase proxy metrics cannot diagnose breath support."),
        blocked("timbre_or_technique_diagnosis", "Tone proxy metrics cannot diagnose timbre, strain, or technique."),
    ]
    blocker_keys = {(item.get("type"), item.get("reason")) for item in blocked_feedback}
    for item in proxy_blockers:
        key = (item.get("type"), item.get("reason"))
        if key not in blocker_keys:
            blocked_feedback.append(item)
            blocker_keys.add(key)
    for caveat in (BREATH_PROXY_CAVEAT, TONE_PROXY_CAVEAT):
        if caveat not in caveats:
            caveats.append(caveat)
    task_result = {
        "task_type": task_type,
        "status": status,
        "score_status": status,
        "full_song_score": full_song_score,
        "diagnostic_score": diagnostic_score,
        "summary": summary,
        "next_exercise_suggestion": next_exercise_suggestion,
    }
    return {
        "task_result": task_result,
        "subscores": subscores,
        "allowed_feedback": allowed_feedback,
        "blocked_feedback": blocked_feedback,
        "caveats": caveats,
        "next_exercise_suggestion": next_exercise_suggestion,
        "feedback_policy": {
            "allowed_feedback": allowed_feedback,
            "blocked_feedback": blocked_feedback,
            "caveats": caveats,
        },
    }


def unsupported_task(task_type: str) -> dict[str, Any]:
    return result(
        task_type=task_type,
        status="unsupported_task",
        full_song_score=None,
        diagnostic_score=None,
        summary=f"Task type {task_type!r} is not implemented by H3.",
        subscores={},
        allowed_feedback=["signal_quality"],
        blocked_feedback=[blocked("task_score", "Unsupported task type.")],
        caveats=["Unsupported task type."],
        next_exercise_suggestion=None,
    )
