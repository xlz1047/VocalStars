"""Interpretable coaching categories derived from selected F0/VAD evidence."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

HOP_S = 0.01


def _safe_float(value: Any, precision: int = 4) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return round(val, precision)


def _blank_category(name: str, caveat: str) -> dict[str, Any]:
    return {
        "status": "not_enough_evidence",
        "score": None,
        "confidence": 0.0,
        "metrics": {},
        "evidence_segments": [],
        "caveats": [caveat],
        "recommended_exercise": None,
        "source": "selected_f0_vad_algorithmic",
    }


def _frames(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    frames = analysis.get("frames") or []
    return frames if isinstance(frames, list) else []


def _f0_and_times(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = []
    times = []
    conf = []
    for idx, frame in enumerate(frames):
        value = frame.get("f0_hz")
        if value is None:
            f0.append(np.nan)
        else:
            try:
                hz = float(value)
                f0.append(hz if 50.0 <= hz <= 1600.0 else np.nan)
            except (TypeError, ValueError):
                f0.append(np.nan)
        times.append(float(frame.get("time_s", idx * HOP_S) or 0.0))
        conf.append(float(frame.get("pitch_confidence") or frame.get("voice_confidence") or 0.0))
    return np.asarray(f0, dtype=np.float64), np.asarray(times, dtype=np.float64), np.asarray(conf, dtype=np.float64)


def _validity_blocks(analysis: dict[str, Any]) -> str | None:
    validity = analysis.get("analysis_validity") or {}
    status = validity.get("status")
    invalid_type = validity.get("invalid_type")
    if status == "invalid":
        return f"Coaching blocked because the input is marked invalid: {invalid_type or 'unknown'}."
    frames = _frames(analysis)
    if not frames:
        return "Coaching blocked because no UI-ready F0/VAD frames are available."
    f0, _, conf = _f0_and_times(frames)
    coverage = float(np.mean(np.isfinite(f0))) if len(f0) else 0.0
    voiced = float(np.mean([bool(frame.get("voiced")) for frame in frames])) if frames else 0.0
    mean_conf = float(np.nanmean(conf)) if len(conf) else 0.0
    if coverage < 0.20 or voiced < 0.20:
        return "Coaching blocked because there is too little reliable singing evidence."
    if mean_conf < 0.15:
        return "Coaching blocked because pitch/VAD confidence is too low."
    return None


def _regions(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if start is not None and ((not value) or idx == len(mask) - 1):
            end = idx if not value else idx + 1
            if end - start >= min_frames:
                out.append((start, end))
            start = None
    return out


def _cents(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    base = float(np.nanmedian(values[valid]))
    if base <= 0:
        return np.asarray([], dtype=np.float64)
    return 1200.0 * np.log2(values[valid] / base)


def _vibrato_window_metrics(f0: np.ndarray, hop_s: float) -> dict[str, float] | None:
    cents = _cents(f0)
    if cents.size < max(20, int(0.45 / hop_s)):
        return None
    centered = cents - np.nanmean(cents)
    extent = float((np.nanpercentile(centered, 95) - np.nanpercentile(centered, 5)) / 2.0)
    if extent < 12.0:
        return None
    spectrum = np.fft.rfft(centered * np.hanning(centered.size))
    freqs = np.fft.rfftfreq(centered.size, d=hop_s)
    band = (freqs >= 3.5) & (freqs <= 8.5)
    if not np.any(band):
        return None
    power = np.abs(spectrum) ** 2
    band_power = power[band]
    if not np.any(band_power):
        return None
    peak_idx = int(np.argmax(band_power))
    rate = float(freqs[band][peak_idx])
    regularity = float(band_power[peak_idx] / max(float(np.sum(power[freqs >= 1.0])), 1e-9))
    return {"rate_hz": rate, "extent_cents": extent, "regularity": regularity}


def _build_vibrato(analysis: dict[str, Any]) -> dict[str, Any]:
    blocked = _validity_blocks(analysis)
    if blocked:
        return _blank_category("vibrato", blocked)
    frames = _frames(analysis)
    f0, times, conf = _f0_and_times(frames)
    valid = np.isfinite(f0)
    regions = _regions(valid, min_frames=60)
    if not regions:
        return _blank_category("vibrato", "No sustained region of at least 0.6 seconds was found.")

    hop_s = float(np.nanmedian(np.diff(times))) if len(times) > 1 else HOP_S
    hop_s = hop_s if np.isfinite(hop_s) and hop_s > 0 else HOP_S
    detections = []
    long_duration = 0.0
    for start, end in regions:
        region_f0 = f0[start:end]
        long_duration += (end - start) * hop_s
        window_frames = max(45, int(0.60 / hop_s))
        step = max(5, int(0.10 / hop_s))
        best = None
        first_start = None
        for w_start in range(start, max(start + 1, end - window_frames + 1), step):
            w_end = min(end, w_start + window_frames)
            metrics = _vibrato_window_metrics(f0[w_start:w_end], hop_s)
            if not metrics:
                continue
            if 4.5 <= metrics["rate_hz"] <= 8.0 and 18.0 <= metrics["extent_cents"] <= 140.0 and metrics["regularity"] >= 0.18:
                if best is None or metrics["regularity"] > best["regularity"]:
                    best = metrics
                if first_start is None:
                    first_start = w_start
        if best:
            onset_delay = max(0.0, (first_start - start) * hop_s) if first_start is not None else None
            detections.append(
                {
                    "start_s": _safe_float(times[start], 3),
                    "end_s": _safe_float(times[end - 1] + hop_s, 3),
                    "duration_s": _safe_float((end - start) * hop_s, 3),
                    "rate_hz": _safe_float(best["rate_hz"], 3),
                    "extent_cents": _safe_float(best["extent_cents"], 2),
                    "regularity": _safe_float(best["regularity"], 3),
                    "onset_delay_s": _safe_float(onset_delay, 3),
                }
            )

    if not detections:
        return {
            **_blank_category("vibrato", "Sustained singing was detected, but vibrato evidence did not pass rate, extent, and regularity checks."),
            "metrics": {
                "sustained_region_count": len(regions),
                "sustained_duration_s": _safe_float(long_duration, 3),
                "vibrato_presence": 0.0,
            },
            "confidence": _safe_float(min(0.45, float(np.nanmean(conf[valid]))), 3) or 0.0,
        }

    rate = float(np.mean([d["rate_hz"] for d in detections if d["rate_hz"] is not None]))
    extent = float(np.mean([d["extent_cents"] for d in detections if d["extent_cents"] is not None]))
    regularity = float(np.mean([d["regularity"] for d in detections if d["regularity"] is not None]))
    coverage = sum(float(d["duration_s"] or 0.0) for d in detections) / max(long_duration, 1e-6)
    rate_score = max(0.0, 1.0 - abs(rate - 6.0) / 2.5)
    extent_score = max(0.0, 1.0 - abs(extent - 55.0) / 70.0)
    score = 100.0 * (0.35 * rate_score + 0.25 * extent_score + 0.25 * min(1.0, regularity / 0.55) + 0.15 * min(1.0, coverage))
    confidence = min(1.0, 0.45 + 0.35 * min(1.0, float(np.nanmean(conf[valid]))) + 0.20 * min(1.0, regularity / 0.45))
    return {
        "status": "complete",
        "score": _safe_float(score, 2),
        "confidence": _safe_float(confidence, 3),
        "metrics": {
            "vibrato_presence": 1.0,
            "mean_rate_hz": _safe_float(rate, 3),
            "mean_extent_cents": _safe_float(extent, 2),
            "mean_regularity": _safe_float(regularity, 3),
            "sustained_region_count": len(regions),
            "sustained_duration_s": _safe_float(long_duration, 3),
            "sustained_note_coverage": _safe_float(coverage, 3),
        },
        "evidence_segments": detections[:5],
        "caveats": ["Algorithmic coaching from selected F0/VAD; this is not a vocal health or technique diagnosis."],
        "recommended_exercise": "Hold a comfortable note for 4 beats, add vibrato after the first beat, and keep the pulse even.",
        "source": "selected_f0_vad_algorithmic",
    }


def _slide_from_breakdown(analysis: dict[str, Any]) -> dict[str, Any] | None:
    breakdown = (analysis.get("subscores") or {}).get("pitch_slide_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = ((analysis.get("task_result") or {}).get("subscores") or {}).get("pitch_slide_breakdown")
    if not isinstance(breakdown, dict):
        return None
    score = breakdown.get("overall")
    confidence = breakdown.get("diagnostic_confidence", 0.0)
    return {
        "status": "complete" if score is not None else "not_enough_evidence",
        "score": _safe_float(score, 2),
        "confidence": _safe_float(confidence, 3) or 0.0,
        "metrics": {
            "direction": breakdown.get("direction"),
            "pitch_range_cents": _safe_float(breakdown.get("pitch_range_cents"), 2),
            "monotonicity": _safe_float(breakdown.get("monotonicity"), 3),
            "smoothness": _safe_float(breakdown.get("smoothness"), 3),
            "dropout_rate": _safe_float(breakdown.get("dropout_rate"), 3),
            "contour_deviation_cents": _safe_float(breakdown.get("contour_mae_cents"), 2),
        },
        "evidence_segments": breakdown.get("evidence_segments") or [],
        "caveats": ["Slide coaching uses pitch-contour evidence and should abstain when F0/VAD confidence is low."],
        "recommended_exercise": "Slide slowly between two nearby notes, keeping the motion continuous and the endpoint centered.",
        "source": "task_pitch_slide_breakdown",
    }


def _build_slide(analysis: dict[str, Any]) -> dict[str, Any]:
    blocked = _validity_blocks(analysis)
    if blocked:
        return _blank_category("slide", blocked)
    from_breakdown = _slide_from_breakdown(analysis)
    if from_breakdown:
        return from_breakdown

    frames = _frames(analysis)
    f0, times, conf = _f0_and_times(frames)
    valid = np.isfinite(f0)
    regions = _regions(valid, min_frames=30)
    if not regions:
        return _blank_category("slide", "No continuous pitch contour of at least 0.3 seconds was found.")
    start, end = max(regions, key=lambda region: region[1] - region[0])
    values = f0[start:end]
    cents = _cents(values)
    if cents.size < 3:
        return _blank_category("slide", "Pitch contour was too sparse to assess slide control.")
    pitch_range = float(np.nanpercentile(cents, 95) - np.nanpercentile(cents, 5))
    if pitch_range < 120.0:
        return _blank_category("slide", "Pitch movement was too small to treat as a slide/glissando.")
    diffs = np.diff(cents)
    signed = np.sign(float(np.nanmedian(diffs))) or 1.0
    monotonicity = float(np.mean((diffs * signed) >= -20.0))
    jump_p95 = float(np.nanpercentile(np.abs(diffs), 95)) if diffs.size else 0.0
    smoothness = max(0.0, min(1.0, 1.0 - jump_p95 / 220.0))
    dropout = 1.0 - ((end - start) / max(len(frames), 1))
    score = 100.0 * (0.45 * monotonicity + 0.35 * smoothness + 0.20 * max(0.0, 1.0 - dropout))
    direction = "up" if signed > 0 else "down"
    confidence = min(1.0, 0.35 + 0.35 * float(np.nanmean(conf[valid])) + 0.30 * max(0.0, 1.0 - dropout))
    return {
        "status": "complete",
        "score": _safe_float(score, 2),
        "confidence": _safe_float(confidence, 3),
        "metrics": {
            "direction": direction,
            "pitch_range_cents": _safe_float(pitch_range, 2),
            "monotonicity": _safe_float(monotonicity, 3),
            "smoothness": _safe_float(smoothness, 3),
            "dropout_rate": _safe_float(dropout, 3),
            "contour_deviation_cents": None,
        },
        "evidence_segments": [
            {
                "start_s": _safe_float(times[start], 3),
                "end_s": _safe_float(times[end - 1] + HOP_S, 3),
                "direction": direction,
                "pitch_range_cents": _safe_float(pitch_range, 2),
            }
        ],
        "caveats": ["Algorithmic slide coaching without a reference target measures control, not target correctness."],
        "recommended_exercise": "Practice the same slide slowly, then repeat it at tempo while keeping the pitch path connected.",
        "source": "selected_f0_vad_algorithmic",
    }


def build_coaching_categories(analysis: dict[str, Any]) -> dict[str, Any]:
    """Return UI-safe non-pitch/VAD coaching categories for VocalStars v1."""
    return {
        "schema_version": "coaching_categories.v1",
        "policy": {
            "selected_user_facing_labels": ["vibrato", "glissando"],
            "hidden_or_report_only_labels": ["mix", "falsetto", "breathy", "pharyngeal"],
            "model_c_gate": {
                "f1_min": 0.85,
                "precision_min": 0.90,
                "recall_min": 0.80,
                "false_positive_rate_max": 0.05,
                "split_requirement": "held_out_singers_only",
            },
            "fail_closed": True,
        },
        "vibrato": _build_vibrato(analysis),
        "slide": _build_slide(analysis),
    }
