#!/usr/bin/env python3
"""Report-only hybrid decision harness for Model A, NanoPitch, and pyin.

This script does not change product behavior. It runs or loads three pitch/VAD
sources on the same WAV, aligns them to a 10 ms grid, and writes agreement
metrics plus task-aware source recommendations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.inference.coach_inference import HOP_LENGTH, HOP_S, SR, VOICED_THRESH  # noqa: E402
from scripts.eval.audit_model_outputs import run_raw_checkpoint  # noqa: E402
from scripts.eval.compare_baseline_outputs import run_pyin_baseline  # noqa: E402
from scripts.eval.evaluate_nanopitch_wav import (  # noqa: E402
    DEFAULT_CHECKPOINT as NANOPITCH_CHECKPOINT,
)
from scripts.eval.evaluate_nanopitch_wav import run_nanopitch  # noqa: E402


SELF_RECORDED = [
    Path("samples/00_silence.wav"),
    Path("samples/01_speaking_voice.wav"),
    Path("samples/03_sustained_aaa.wav"),
    Path("samples/04_pitch_slide.wav"),
    Path("samples/05_twinkle_twinkle.wav"),
]

SYNTHETIC = [
    Path("samples/synthetic_model_tests/digital_silence_5s.wav"),
    Path("samples/synthetic_model_tests/white_noise_5s.wav"),
    Path("samples/synthetic_model_tests/low_hum_80hz_5s.wav"),
    Path("samples/synthetic_model_tests/sine_220hz_5s.wav"),
    Path("samples/synthetic_model_tests/sine_440hz_5s.wav"),
    Path("samples/synthetic_model_tests/sine_sweep_220_to_440_5s.wav"),
    Path("samples/synthetic_model_tests/pulsed_220hz_voiced_unvoiced.wav"),
]


@dataclass
class SourceTrack:
    name: str
    f0_hz: np.ndarray
    voiced: np.ndarray
    voice_confidence: np.ndarray | None
    pitch_confidence: np.ndarray | None
    runtime_s: float | None = None

    def trim(self, n: int) -> "SourceTrack":
        return SourceTrack(
            name=self.name,
            f0_hz=np.asarray(self.f0_hz[:n], dtype=np.float32),
            voiced=np.asarray(self.voiced[:n], dtype=bool),
            voice_confidence=(
                np.asarray(self.voice_confidence[:n], dtype=np.float32)
                if self.voice_confidence is not None
                else None
            ),
            pitch_confidence=(
                np.asarray(self.pitch_confidence[:n], dtype=np.float32)
                if self.pitch_confidence is not None
                else None
            ),
            runtime_s=self.runtime_s,
        )


def load_audio(path: Path) -> np.ndarray:
    from ml_new.inference.coach_inference import _load_audio_robust, SR as _SR

    audio = _load_audio_robust(path, sr=_SR)
    min_samples = _SR
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant")
    return audio


def array_summary(values: np.ndarray | None) -> dict[str, Any]:
    if values is None:
        return {"mean": None, "median": None, "min": None, "max": None, "percentiles": {}}
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "percentiles": {}}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {
            f"p{p:02d}": float(np.percentile(arr, p))
            for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    }


def load_model_a_track(audio_path: Path, audio: np.ndarray, checkpoint: Path, device: str) -> SourceTrack:
    sample = audio_path.stem
    m0_npz = REPO_ROOT / "reports" / "model_output_audit" / sample / f"{sample}_raw_outputs.npz"
    if m0_npz.exists():
        data = np.load(m0_npz)
        f0 = data["raw_f0_voiced_thresholded_hz"].astype(np.float32)
        voiced_prob = data["voiced_prob"].astype(np.float32)
        pitch_conf = data["pitch_confidence"].astype(np.float32)
    else:
        raw = run_raw_checkpoint(audio, checkpoint, device)
        f0 = raw["raw_f0_voiced_thresholded_hz"].astype(np.float32)
        voiced_prob = raw["voiced_prob"].astype(np.float32)
        pitch_conf = raw["pitch_confidence"].astype(np.float32)
    return SourceTrack(
        name="model_a",
        f0_hz=f0,
        voiced=voiced_prob >= VOICED_THRESH,
        voice_confidence=voiced_prob,
        pitch_confidence=pitch_conf,
    )


def load_nanopitch_track(audio_path: Path, audio: np.ndarray, checkpoint: Path, device: str) -> SourceTrack:
    sample = audio_path.stem
    np_json = REPO_ROOT / "reports" / "nanopitch_eval" / sample / f"{sample}_nanopitch.json"
    if np_json.exists():
        data = json.loads(np_json.read_text(encoding="utf-8"))
        f0 = np.asarray(data["f0_hz"], dtype=np.float32)
        vad_prob = np.asarray(data["vad_prob"], dtype=np.float32)
        voiced = np.asarray(data["voiced_mask"], dtype=bool)
        pitch_conf = np.asarray(data["pitch_confidence"], dtype=np.float32)
        runtime_s = data.get("runtime_s")
    else:
        raw = run_nanopitch(audio, checkpoint, device)
        f0 = np.asarray(raw["f0_hz"], dtype=np.float32)
        vad_prob = np.asarray(raw["vad_prob"], dtype=np.float32)
        voiced = vad_prob >= 0.5
        pitch_conf = np.asarray(raw["pitch_confidence"], dtype=np.float32)
        runtime_s = raw.get("runtime_s")
    return SourceTrack(
        name="nanopitch",
        f0_hz=f0,
        voiced=voiced,
        voice_confidence=vad_prob,
        pitch_confidence=pitch_conf,
        runtime_s=float(runtime_s) if runtime_s is not None else None,
    )


def load_pyin_track(audio: np.ndarray) -> SourceTrack:
    raw = run_pyin_baseline(audio)
    f0 = np.asarray(raw["f0_hz"], dtype=np.float32)
    voiced = np.asarray(raw["voiced_mask"], dtype=bool)
    voiced_prob = np.asarray(raw["voiced_prob"], dtype=np.float32)
    return SourceTrack(
        name="pyin",
        f0_hz=f0,
        voiced=voiced,
        voice_confidence=voiced_prob,
        pitch_confidence=voiced_prob,
    )


def align_tracks(tracks: list[SourceTrack]) -> list[SourceTrack]:
    n = min(len(t.f0_hz) for t in tracks)
    return [t.trim(n) for t in tracks]


def pct(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def cents_between(a_hz: np.ndarray, b_hz: np.ndarray) -> np.ndarray:
    a = np.asarray(a_hz, dtype=np.float64)
    b = np.asarray(b_hz, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out = np.full_like(a, np.nan, dtype=np.float64)
    out[valid] = 1200.0 * np.log2(a[valid] / b[valid])
    return out


def jump_counts(f0_hz: np.ndarray) -> dict[str, int]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    octave = 0
    semitone = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        cents = abs(1200.0 * math.log2(float(f0[b]) / max(float(f0[a]), 1e-9)))
        if cents >= 900.0:
            octave += 1
        if cents >= 200.0:
            semitone += 1
    return {"octave_jump_count": int(octave), "semitone_jump_count": int(semitone)}


def source_summary(track: SourceTrack) -> dict[str, Any]:
    valid_f0 = np.isfinite(track.f0_hz) & (track.f0_hz > 0)
    f0_vals = track.f0_hz[valid_f0]
    jumps = jump_counts(track.f0_hz)
    return {
        "f0_coverage": float(np.mean(valid_f0)) if len(track.f0_hz) else 0.0,
        "voiced_percentage": pct(track.voiced),
        "median_f0_hz": float(np.median(f0_vals)) if f0_vals.size else None,
        "min_f0_hz": float(np.min(f0_vals)) if f0_vals.size else None,
        "max_f0_hz": float(np.max(f0_vals)) if f0_vals.size else None,
        "octave_jump_count": jumps["octave_jump_count"],
        "semitone_jump_count": jumps["semitone_jump_count"],
        "voice_confidence": array_summary(track.voice_confidence),
        "pitch_confidence": array_summary(track.pitch_confidence),
        "runtime_s": track.runtime_s,
    }


def pairwise_f0_agreement(a: SourceTrack, b: SourceTrack) -> dict[str, Any]:
    overlap = a.voiced & b.voiced & (a.f0_hz > 0) & (b.f0_hz > 0)
    cents = np.abs(cents_between(a.f0_hz, b.f0_hz))
    vals = cents[overlap & np.isfinite(cents)]
    if vals.size == 0:
        return {
            "overlap_frame_percentage": 0.0,
            "median_abs_cents": None,
            "mean_abs_cents": None,
            "frames_gt_50_cents": None,
            "frames_gt_100_cents": None,
            "frames_gt_200_cents": None,
            "octave_mismatch_rate": None,
        }
    return {
        "overlap_frame_percentage": pct(overlap),
        "median_abs_cents": float(np.median(vals)),
        "mean_abs_cents": float(np.mean(vals)),
        "frames_gt_50_cents": float(np.mean(vals > 50.0)),
        "frames_gt_100_cents": float(np.mean(vals > 100.0)),
        "frames_gt_200_cents": float(np.mean(vals > 200.0)),
        "octave_mismatch_rate": float(np.mean(vals >= 900.0)),
    }


def multi_source_f0_disagreement(tracks: dict[str, SourceTrack]) -> dict[str, Any]:
    names = list(tracks)
    n = len(next(iter(tracks.values())).f0_hz)
    disagreement = []
    for i in range(n):
        vals = []
        for name in names:
            t = tracks[name]
            if t.voiced[i] and np.isfinite(t.f0_hz[i]) and t.f0_hz[i] > 0:
                vals.append(float(t.f0_hz[i]))
        if len(vals) < 2:
            continue
        cents = 1200.0 * np.log2(np.asarray(vals) / max(float(np.median(vals)), 1e-9))
        disagreement.append(float(np.max(cents) - np.min(cents)))
    arr = np.asarray(disagreement, dtype=np.float64)
    if arr.size == 0:
        return {
            "overlap_frame_percentage": 0.0,
            "median_spread_cents": None,
            "mean_spread_cents": None,
            "frames_gt_50_cents": None,
            "frames_gt_100_cents": None,
            "frames_gt_200_cents": None,
            "octave_mismatch_rate": None,
        }
    return {
        "overlap_frame_percentage": float(arr.size / max(n, 1)),
        "median_spread_cents": float(np.median(arr)),
        "mean_spread_cents": float(np.mean(arr)),
        "frames_gt_50_cents": float(np.mean(arr > 50.0)),
        "frames_gt_100_cents": float(np.mean(arr > 100.0)),
        "frames_gt_200_cents": float(np.mean(arr > 200.0)),
        "octave_mismatch_rate": float(np.mean(arr >= 900.0)),
    }


def infer_task_type(sample: str) -> str:
    if "speaking" in sample or "speech" in sample:
        return "speech_like_or_non_singing_test"
    if "sustained" in sample:
        return "sustained_note"
    if "pitch_slide" in sample or "sweep" in sample:
        return "pitch_slide"
    if "twinkle" in sample:
        return "free_singing"
    if "silence" in sample or "noise" in sample or "hum" in sample:
        return "invalid_or_signal_quality_test"
    if "sine_" in sample or "pulsed" in sample:
        return "synthetic_pitch_test"
    return "free_singing"


def task_type_from_config(task_config: dict[str, Any] | None) -> str | None:
    if not isinstance(task_config, dict):
        return None
    task_type = task_config.get("task_type")
    if isinstance(task_type, str) and task_type.strip():
        return task_type.strip()
    return None


def recommendation(
    sample: str,
    tracks: dict[str, SourceTrack],
    metrics: dict[str, Any],
    *,
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_type = task_type_from_config(task_config) or infer_task_type(sample)
    source_summaries = metrics["sources"]
    agreement = metrics["voiced_agreement"]
    f0_multi = metrics["f0_disagreement"]
    reason_codes: list[str] = []

    model_a_cov = source_summaries["model_a"]["f0_coverage"]
    nano_cov = source_summaries["nanopitch"]["f0_coverage"]
    pyin_cov = source_summaries["pyin"]["f0_coverage"]
    model_a_voiced = source_summaries["model_a"]["voiced_percentage"]
    nano_voiced = source_summaries["nanopitch"]["voiced_percentage"]
    pyin_voiced = source_summaries["pyin"]["voiced_percentage"]
    model_a_jumps = (
        source_summaries["model_a"]["octave_jump_count"]
        + source_summaries["model_a"]["semitone_jump_count"]
    )
    pyin_jumps = (
        source_summaries["pyin"]["octave_jump_count"]
        + source_summaries["pyin"]["semitone_jump_count"]
    )

    signal_quality = "usable"
    selected_vad = "hybrid"
    selected_f0 = "hybrid"

    if agreement["no_sources_voiced_percentage"] > 0.95:
        signal_quality = "no_voice_or_rejected_by_all_sources"
        selected_vad = "none"
        selected_f0 = "none"
        reason_codes.append("no_sources_voiced")
    elif nano_voiced < 0.02 and model_a_voiced > 0.3 and pyin_voiced > 0.3:
        signal_quality = "possible_noise_or_non_singing_disagreement"
        selected_vad = "nanopitch_guard"
        selected_f0 = "none"
        reason_codes.append("nanopitch_negative_model_a_pyin_positive")
    elif f0_multi["frames_gt_200_cents"] is not None and f0_multi["frames_gt_200_cents"] > 0.25:
        signal_quality = "strong_f0_source_disagreement"
        reason_codes.append("f0_disagreement_gt_200_cents")

    if task_type == "sustained_note":
        if pyin_cov > 0.5 and pyin_jumps <= 2:
            selected_f0 = "pyin"
            reason_codes.append("sustained_note_prefers_stable_pyin")
        elif nano_cov > 0.5:
            selected_f0 = "nanopitch"
            reason_codes.append("sustained_note_nanopitch_has_coverage")
        else:
            selected_f0 = "none"
            reason_codes.append("sustained_note_no_reliable_f0_source")
    elif task_type == "pitch_slide":
        if pyin_cov > 0.5 and pyin_jumps <= 3:
            selected_f0 = "pyin"
            reason_codes.append("pitch_slide_prefers_pyin_contour")
        elif nano_cov > 0.5:
            selected_f0 = "nanopitch"
            reason_codes.append("pitch_slide_nanopitch_available")
        else:
            selected_f0 = "none"
            reason_codes.append("pitch_slide_no_reliable_f0_source")
    elif task_type == "free_singing":
        if model_a_cov > 0.6:
            selected_f0 = "model_a_with_pyin_guard"
            reason_codes.append("free_singing_uses_model_a_coverage")
            if pyin_cov > 0.5:
                reason_codes.append("pyin_available_as_f0_guard")
        elif pyin_cov > 0.5:
            selected_f0 = "pyin"
            reason_codes.append("free_singing_model_a_low_coverage_pyin_available")
        else:
            selected_f0 = "none"
            reason_codes.append("free_singing_no_reliable_f0_source")
    elif task_type == "speech_like_or_non_singing_test":
        selected_vad = "nanopitch_guard"
        selected_f0 = "none"
        signal_quality = "speech_like_or_non_singing_candidate"
        reason_codes.append("speech_like_fixture_no_singing_f0_recommendation")
        if nano_voiced < 0.02:
            reason_codes.append("nanopitch_rejects_speech_like_input")
    elif task_type == "invalid_or_signal_quality_test":
        if nano_voiced < 0.02:
            selected_vad = "nanopitch_guard"
            selected_f0 = "none"
            reason_codes.append("invalid_signal_nanopitch_rejects")
        elif pyin_cov > 0.5 and model_a_cov > 0.5:
            selected_f0 = "none"
            signal_quality = "likely_tonal_non_voice"
            reason_codes.append("invalid_signal_tonal_sources_positive")
    elif task_type == "synthetic_pitch_test":
        if pyin_cov > 0.5 and pyin_jumps <= model_a_jumps:
            selected_f0 = "pyin"
            reason_codes.append("synthetic_pitch_prefers_pyin")
        elif model_a_cov > 0.5:
            selected_f0 = "model_a"
            reason_codes.append("synthetic_pitch_model_a_available")
        else:
            selected_f0 = "none"
            reason_codes.append("synthetic_pitch_no_reliable_source")

    if selected_vad == "hybrid":
        if nano_voiced > 0.5 and (model_a_voiced > 0.5 or pyin_voiced > 0.5):
            selected_vad = "nanopitch_plus_model_or_pyin"
            reason_codes.append("nanopitch_positive_with_support")
        elif model_a_voiced > 0.5 and pyin_voiced > 0.5:
            selected_vad = "model_a_plus_pyin_with_nanopitch_caveat"
            reason_codes.append("model_a_pyin_positive_nanopitch_caveat")
        else:
            selected_vad = "none"
            reason_codes.append("insufficient_vad_agreement")

    return {
        "task_type": task_type,
        "selected_f0_source_recommendation": selected_f0,
        "selected_vad_source_recommendation": selected_vad,
        "signal_quality_status": signal_quality,
        "reason_codes": sorted(set(reason_codes)),
    }


def compute_metrics(
    audio_path: Path,
    tracks_list: list[SourceTrack],
    *,
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracks = {t.name: t for t in tracks_list}
    m = tracks["model_a"].voiced
    n = tracks["nanopitch"].voiced
    p = tracks["pyin"].voiced
    all_sources = m & n & p
    no_sources = ~(m | n | p)
    exactly_model = m & ~n & ~p
    exactly_nano = n & ~m & ~p
    exactly_pyin = p & ~m & ~n
    all_same = all_sources | no_sources

    pairwise = {
        "model_a_vs_nanopitch": pairwise_f0_agreement(tracks["model_a"], tracks["nanopitch"]),
        "model_a_vs_pyin": pairwise_f0_agreement(tracks["model_a"], tracks["pyin"]),
        "nanopitch_vs_pyin": pairwise_f0_agreement(tracks["nanopitch"], tracks["pyin"]),
    }
    metrics = {
        "sample": audio_path.stem,
        "input_path": str(audio_path),
        "frames": int(len(m)),
        "hop_s": HOP_S,
        "duration_s": float(len(m) * HOP_S),
        "voiced_agreement": {
            "voiced_agreement_percentage": pct(all_same),
            "model_a_only_voiced_percentage": pct(exactly_model),
            "nanopitch_only_voiced_percentage": pct(exactly_nano),
            "pyin_only_voiced_percentage": pct(exactly_pyin),
            "all_sources_voiced_percentage": pct(all_sources),
            "no_sources_voiced_percentage": pct(no_sources),
        },
        "pairwise_f0_agreement": pairwise,
        "f0_disagreement": multi_source_f0_disagreement(tracks),
        "sources": {name: source_summary(track) for name, track in tracks.items()},
    }
    metrics["recommendation"] = recommendation(
        audio_path.stem,
        tracks,
        metrics,
        task_config=task_config,
    )
    return metrics


def write_svg(path: Path, audio: np.ndarray, tracks_list: list[SourceTrack], title: str) -> None:
    width = 1200
    height = 760
    left = 82
    right = 32
    plot_w = width - left - right
    duration = max(len(audio) / SR, HOP_S)
    tracks = {t.name: t for t in tracks_list}
    T = min(len(t.f0_hz) for t in tracks_list)
    times = np.arange(T) * HOP_S

    def x_at(t: float) -> float:
        return left + (t / duration) * plot_w

    def polyline(points: list[tuple[float, float]], color: str, width_px: float = 1.4) -> str:
        if not points:
            return ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" />'

    def f0_points(f0: np.ndarray, top: float, bottom: float, fmin: float, fmax: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for t, hz in zip(times, f0[:T]):
            if not np.isfinite(hz) or hz <= 0:
                continue
            frac = (math.log2(float(hz)) - math.log2(fmin)) / max(
                math.log2(fmax) - math.log2(fmin), 1e-9
            )
            y = bottom - float(np.clip(frac, 0.0, 1.0)) * (bottom - top)
            pts.append((x_at(float(t)), y))
        return pts

    valid_f0 = np.concatenate([t.f0_hz[np.isfinite(t.f0_hz) & (t.f0_hz > 0)] for t in tracks_list])
    if valid_f0.size:
        fmin = max(30.0, float(np.percentile(valid_f0, 1)) * 0.8)
        fmax = min(2200.0, float(np.percentile(valid_f0, 99)) * 1.2)
        if fmax <= fmin:
            fmin, fmax = 50.0, 800.0
    else:
        fmin, fmax = 50.0, 800.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">{escape(title)}</text>',
    ]

    rows = [
        ("Waveform", 52, 145),
        ("F0 comparison", 195, 430),
        ("Voiced masks", 500, 700),
    ]
    for label, top, bottom in rows:
        svg.append(f'<text x="18" y="{top + 18}" font-size="13" font-family="Arial">{label}</text>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="#ddd" />')

    # Waveform envelope.
    row_top, row_bottom = rows[0][1], rows[0][2]
    if audio.size:
        bins = min(900, max(1, audio.size // 120))
        step = max(1, audio.size // bins)
        chunks = audio[: bins * step].reshape(bins, step)
        peak = np.max(np.abs(chunks), axis=1)
        max_peak = max(float(np.max(peak)), 1e-9)
        mid = (row_top + row_bottom) / 2
        scale = (row_bottom - row_top) / 2
        pts_top = []
        pts_bottom = []
        for i, val in enumerate(peak):
            t = (i * step) / SR
            x = x_at(t)
            amp = float(val) / max_peak * scale
            pts_top.append((x, mid - amp))
            pts_bottom.append((x, mid + amp))
        area = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts_top + list(reversed(pts_bottom)))
        svg.append(f'<polygon points="{area}" fill="#dbeafe" stroke="#60a5fa" stroke-width="1" />')

    # F0.
    top, bottom = rows[1][1], rows[1][2]
    colors = {"model_a": "#ef4444", "nanopitch": "#7c3aed", "pyin": "#059669"}
    labels = {"model_a": "Model A", "nanopitch": "NanoPitch", "pyin": "pyin"}
    for name, track in tracks.items():
        svg.append(polyline(f0_points(track.f0_hz, top, bottom, fmin, fmax), colors[name], 1.8))
    svg.append(f'<text x="{left}" y="{top - 8}" font-size="12" font-family="Arial" fill="#555">{fmax:.1f} Hz</text>')
    svg.append(f'<text x="{left}" y="{bottom + 16}" font-size="12" font-family="Arial" fill="#555">{fmin:.1f} Hz</text>')
    lx = left + 12
    for i, name in enumerate(("model_a", "nanopitch", "pyin")):
        y = top + 18 + i * 20
        svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx+28}" y2="{y}" stroke="{colors[name]}" stroke-width="3" />')
        svg.append(f'<text x="{lx+36}" y="{y+4}" font-size="12" font-family="Arial">{labels[name]}</text>')

    # Voiced masks.
    top, bottom = rows[2][1], rows[2][2]
    lane_h = 34
    for i, name in enumerate(("model_a", "nanopitch", "pyin")):
        y = top + 18 + i * 54
        svg.append(f'<text x="{left}" y="{y-5}" font-size="12" font-family="Arial">{labels[name]}</text>')
        svg.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#eee" />')
        mask = tracks[name].voiced[:T]
        start = None
        for idx, val in enumerate(mask):
            if val and start is None:
                start = idx
            if start is not None and ((not val) or idx == len(mask) - 1):
                end = idx if not val else idx + 1
                x1 = x_at(start * HOP_S)
                x2 = x_at(end * HOP_S)
                svg.append(
                    f'<rect x="{x1:.2f}" y="{y}" width="{max(x2-x1, 1):.2f}" height="{lane_h}" fill="{colors[name]}" opacity="0.45" />'
                )
                start = None

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_sample_markdown(path: Path, result: dict[str, Any]) -> None:
    rec = result["recommendation"]
    lines = [
        f"# Hybrid Decision: {result['sample']}",
        "",
        f"- Input: `{result['input_path']}`",
        f"- Frames: `{result['frames']}` at `{result['hop_s']}` s",
        f"- Task type inferred for recommendation: `{rec['task_type']}`",
        f"- Selected f0 source recommendation: `{rec['selected_f0_source_recommendation']}`",
        f"- Selected VAD source recommendation: `{rec['selected_vad_source_recommendation']}`",
        f"- Signal quality status: `{rec['signal_quality_status']}`",
        f"- Reason codes: `{', '.join(rec['reason_codes'])}`",
        "",
        "## Voiced Agreement",
        "",
    ]
    for key, value in result["voiced_agreement"].items():
        lines.append(f"- `{key}`: `{100.0 * value:.1f}%`")
    lines += ["", "## Source Summaries", "", "| Source | F0 coverage | Voiced | Median f0 | Oct jumps | Semitone jumps | Pitch conf median |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in result["sources"].items():
        lines.append(
            "| {name} | {f0_cov:.1f}% | {voiced:.1f}% | {median} | {octave} | {semi} | {conf} |".format(
                name=name,
                f0_cov=100.0 * summary["f0_coverage"],
                voiced=100.0 * summary["voiced_percentage"],
                median=fmt(summary["median_f0_hz"]),
                octave=summary["octave_jump_count"],
                semi=summary["semitone_jump_count"],
                conf=fmt(summary["pitch_confidence"]["median"]),
            )
        )
    lines += ["", "## F0 Disagreement", ""]
    for key, value in result["f0_disagreement"].items():
        lines.append(f"- `{key}`: `{fmt_pct_or_value(value, key)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def fmt_pct_or_value(value: Any, key: str) -> str:
    if value is None:
        return "null"
    if key.endswith("percentage") or key.startswith("frames_gt") or key.endswith("rate"):
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def process_file(audio_path: Path, output_dir: Path, checkpoint: Path, nanopitch_checkpoint: Path, device: str) -> dict[str, Any]:
    audio_path = audio_path.resolve()
    audio = load_audio(audio_path)
    tracks = align_tracks(
        [
            load_model_a_track(audio_path, audio, checkpoint, device),
            load_nanopitch_track(audio_path, audio, nanopitch_checkpoint, device),
            load_pyin_track(audio),
        ]
    )
    result = compute_metrics(audio_path, tracks)
    rel = audio_path.stem
    sample_dir = output_dir / rel
    sample_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "json": str(sample_dir / f"{rel}_hybrid_decision.json"),
        "markdown": str(sample_dir / f"{rel}_hybrid_decision.md"),
        "plot": str(sample_dir / f"{rel}_hybrid_decision.svg"),
    }
    result["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_sample_markdown(Path(artifacts["markdown"]), result)
    write_svg(Path(artifacts["plot"]), audio, tracks, f"Hybrid decision: {rel}")
    return result


def write_overall_report(path: Path, results: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# H1 Hybrid Decision Report",
        "",
        "Report-only hybrid decision harness for Model A, NanoPitch, and pyin.",
        "",
        "No user-facing scoring, model architecture, training, frontend behavior, or P4 regression expectations were changed.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        "- Per-file JSON, markdown, and SVG plots are written under `reports/hybrid_decision/<sample>/`.",
        "",
        "## Summary",
        "",
        "| Sample | Task | F0 recommendation | VAD recommendation | Signal quality | All voiced | No voiced | Model A only | NanoPitch only | pyin only | Reason codes |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        rec = r["recommendation"]
        va = r["voiced_agreement"]
        lines.append(
            "| {sample} | `{task}` | `{f0}` | `{vad}` | `{quality}` | {allv:.1f}% | {none:.1f}% | {ma:.1f}% | {np:.1f}% | {py:.1f}% | `{reasons}` |".format(
                sample=f"`{r['sample']}`",
                task=rec["task_type"],
                f0=rec["selected_f0_source_recommendation"],
                vad=rec["selected_vad_source_recommendation"],
                quality=rec["signal_quality_status"],
                allv=100.0 * va["all_sources_voiced_percentage"],
                none=100.0 * va["no_sources_voiced_percentage"],
                ma=100.0 * va["model_a_only_voiced_percentage"],
                np=100.0 * va["nanopitch_only_voiced_percentage"],
                py=100.0 * va["pyin_only_voiced_percentage"],
                reasons=", ".join(rec["reason_codes"]),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- NanoPitch negative VAD is useful as a no-voice/noise warning, especially when Model A or pyin alone marks frames voiced.",
        "- pyin is recommended for sustained-note and pitch-slide f0 when it has adequate coverage and low jumps.",
        "- Model A remains useful for free-singing coverage and app-specific heads, but f0 should be guarded by pyin agreement and diagnostics.",
        "- When sources disagree strongly, the harness recommends `none` or emits source-disagreement reason codes instead of pretending one source is authoritative.",
        "",
        "## Caveats",
        "",
        "- This harness is report-only and does not alter product scoring or coaching.",
        "- Source confidences are not calibrated against each other.",
        "- NanoPitch f0 can be clean when present but sparse on some real singing and synthetic tones.",
        "- pyin can false-voice noise and is not a singing-validity classifier.",
        "- Model A is the only current source for app-specific non-f0 heads.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hybrid_decision"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--nanopitch-checkpoint", type=Path, default=NANOPITCH_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Model A checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.nanopitch_checkpoint.exists():
        print(f"NanoPitch checkpoint not found: {args.nanopitch_checkpoint}", file=sys.stderr)
        return 2

    files = args.files if args.files else SELF_RECORDED + SYNTHETIC
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Evaluating {path}")
        results.append(
            process_file(
                path,
                args.output_dir,
                args.checkpoint,
                args.nanopitch_checkpoint,
                args.device,
            )
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_overall_report(Path("H1_HYBRID_DECISION_REPORT.md"), results, args.output_dir)
    print(json.dumps({"status": "complete", "summary": str(summary_path), "report": "H1_HYBRID_DECISION_REPORT.md"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
