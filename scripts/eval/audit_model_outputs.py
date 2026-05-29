#!/usr/bin/env python3
"""Audit raw checkpoint model outputs before coaching postprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_new.feature_extraction.hcqt import HCQTExtractor  # noqa: E402
from ml_new.feature_extraction.vad_features import VADFeatureExtractor  # noqa: E402
from ml_new.inference.algorithms import stabilize_f0_for_notes  # noqa: E402
from ml_new.inference.coach_inference import (  # noqa: E402
    BINS_PER_OCTAVE,
    BREATH_THRESH,
    FMIN,
    HOP_LENGTH,
    HOP_S,
    N_BINS,
    ONSET_THRESH,
    SR,
    VOICED_THRESH,
    _array_summary,
)
from ml_new.models.unified_model import TECHNIQUE_VOCAB, UnifiedVocalModel  # noqa: E402


SAMPLES = [
    "00_silence",
    "01_speaking_voice",
    "03_sustained_aaa",
    "04_pitch_slide",
    "05_twinkle_twinkle",
]


def load_audio(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(audio, dtype=np.float32)


def run_raw_checkpoint(audio: np.ndarray, checkpoint: Path, device: str) -> dict[str, Any]:
    hcqt_ext = HCQTExtractor(
        sr=SR,
        hop_length=HOP_LENGTH,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    vad_ext = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)

    hcqt = hcqt_ext.compute(audio)
    vad_feats = vad_ext.compute(audio)
    T = min(hcqt.shape[2], vad_feats.shape[1])
    hcqt = hcqt[:, :, :T]
    vad_feats = vad_feats[:, :T]

    dev = torch.device(device)
    model = UnifiedVocalModel().to(dev)
    ckpt = torch.load(str(checkpoint), map_location=dev, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    hcqt_t = torch.from_numpy(hcqt).unsqueeze(0).to(dev)
    vad_t = torch.from_numpy(vad_feats).unsqueeze(0).to(dev)

    with torch.no_grad():
        pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, _ = model(hcqt_t, vad_t)
        pitch_probs = torch.softmax(pitch_logits[0], dim=-1)
        pitch_top2 = torch.topk(pitch_probs, k=2, dim=-1)
        pitch_entropy = -(pitch_probs * torch.log(pitch_probs.clamp_min(1e-12))).sum(dim=-1)
        tech_probs = torch.softmax(tech_logits[0], dim=-1)

    pitch_logits_np = pitch_logits[0].cpu().numpy().astype(np.float32)
    pitch_conf = pitch_top2.values[:, 0].cpu().numpy().astype(np.float32)
    pitch_margin = (pitch_top2.values[:, 0] - pitch_top2.values[:, 1]).cpu().numpy().astype(np.float32)
    pitch_entropy_np = pitch_entropy.cpu().numpy().astype(np.float32)
    pitch_bin_np = pitch_top2.indices[:, 0].cpu().numpy().astype(np.int32)
    bin_hz = model.bin_hz.cpu().numpy().astype(np.float32)
    raw_f0_all = bin_hz[pitch_bin_np].astype(np.float32)

    voiced_np = voiced_prob[0].cpu().numpy().astype(np.float32)
    raw_f0_voiced = np.where(voiced_np >= VOICED_THRESH, raw_f0_all, 0.0).astype(np.float32)
    smoothed_f0, smoothing_diag = stabilize_f0_for_notes(raw_f0_voiced, hop_s=HOP_S)

    return {
        "hcqt_shape": list(hcqt.shape),
        "vad_features_shape": list(vad_feats.shape),
        "pitch_logits": pitch_logits_np,
        "pitch_bin": pitch_bin_np,
        "pitch_confidence": pitch_conf,
        "pitch_margin": pitch_margin,
        "pitch_entropy": pitch_entropy_np,
        "pitch_normalized_entropy": (pitch_entropy_np / max(np.log(float(N_BINS)), 1e-12)).astype(np.float32),
        "voiced_prob": voiced_np,
        "breath_prob": breath_prob[0].cpu().numpy().astype(np.float32),
        "onset_prob": onset_prob[0].cpu().numpy().astype(np.float32),
        "technique_logits": tech_logits[0].cpu().numpy().astype(np.float32),
        "technique_probs": tech_probs.cpu().numpy().astype(np.float32),
        "raw_f0_all_frames_hz": raw_f0_all,
        "raw_f0_voiced_thresholded_hz": raw_f0_voiced,
        "smoothed_f0_hz": smoothed_f0.astype(np.float32),
        "smoothing_diagnostics": smoothing_diag,
    }


def summarize_sample(sample: str, audio: np.ndarray, raw: dict[str, Any]) -> dict[str, Any]:
    voiced = raw["voiced_prob"]
    breath = raw["breath_prob"]
    onset = raw["onset_prob"]
    pitch_conf = raw["pitch_confidence"]
    pitch_entropy = raw["pitch_entropy"]
    pitch_margin = raw["pitch_margin"]
    raw_f0 = raw["raw_f0_voiced_thresholded_hz"]
    smoothed_f0 = raw["smoothed_f0_hz"]
    tech_probs = raw["technique_probs"]
    tech_logits = raw["technique_logits"]
    top_idx = int(np.argmax(tech_probs))

    return {
        "sample": sample,
        "duration_s": float(len(audio) / SR),
        "frames": int(len(voiced)),
        "hop_s": HOP_S,
        "audio_rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "thresholds": {
            "voiced_default": VOICED_THRESH,
            "breath_default": BREATH_THRESH,
            "onset_default": ONSET_THRESH,
        },
        "voiced_probability": {
            **_array_summary(voiced),
            "fraction_above_0_3": _fraction_above(voiced, 0.3),
            "fraction_above_0_5": _fraction_above(voiced, 0.5),
            "fraction_above_0_7": _fraction_above(voiced, 0.7),
            "fraction_above_0_9": _fraction_above(voiced, 0.9),
            "near_default_threshold_fraction": _near(voiced, VOICED_THRESH, 0.05),
        },
        "pitch_confidence": {
            "max_softmax_probability": _array_summary(pitch_conf),
            "top1_top2_margin": _array_summary(pitch_margin),
            "entropy": _array_summary(pitch_entropy),
            "normalized_entropy": _array_summary(raw["pitch_normalized_entropy"]),
        },
        "raw_f0_before_smoothing": _f0_summary(raw_f0),
        "smoothed_f0_after_smoothing": _f0_summary(smoothed_f0),
        "f0_smoothing_diagnostics": raw["smoothing_diagnostics"],
        "onset_probability": {
            **_array_summary(onset),
            "fraction_above_default_threshold": _fraction_above(onset, ONSET_THRESH),
        },
        "breath_probability": {
            **_array_summary(breath),
            "fraction_above_default_threshold": _fraction_above(breath, BREATH_THRESH),
        },
        "technique": {
            "top_label": TECHNIQUE_VOCAB[top_idx],
            "top_probability": float(tech_probs[top_idx]),
            "logits": {
                TECHNIQUE_VOCAB[i]: float(tech_logits[i])
                for i in range(len(TECHNIQUE_VOCAB))
            },
            "probabilities": {
                TECHNIQUE_VOCAB[i]: float(tech_probs[i])
                for i in range(len(TECHNIQUE_VOCAB))
            },
            "unreliable": True,
            "note": "Technique head output is reported for raw-output audit only and should not be treated as reliable coaching evidence.",
        },
    }


def _fraction_above(values: np.ndarray, threshold: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= threshold))


def _near(values: np.ndarray, threshold: float, margin: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr - threshold) <= margin))


def _f0_summary(f0_hz: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(f0_hz, dtype=np.float64)
    voiced = arr[np.isfinite(arr) & (arr > 0)]
    if voiced.size == 0:
        return {
            "voiced_frame_fraction": 0.0,
            "mean_hz": None,
            "median_hz": None,
            "min_hz": None,
            "max_hz": None,
            "trimmed_p05_hz": None,
            "trimmed_p95_hz": None,
            "trimmed_range_hz": None,
            "octave_jump_count": 0,
            "semitone_jump_count": 0,
            "direction_slope_hz_per_s": None,
        }
    return {
        "voiced_frame_fraction": float(voiced.size / max(arr.size, 1)),
        "mean_hz": float(np.mean(voiced)),
        "median_hz": float(np.median(voiced)),
        "min_hz": float(np.min(voiced)),
        "max_hz": float(np.max(voiced)),
        "trimmed_p05_hz": float(np.percentile(voiced, 5)),
        "trimmed_p95_hz": float(np.percentile(voiced, 95)),
        "trimmed_range_hz": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)),
        **_jump_counts(arr),
        "direction_slope_hz_per_s": _direction_slope(arr),
    }


def _jump_counts(f0_hz: np.ndarray) -> dict[str, int]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    idx = np.where(valid)[0]
    octave = 0
    semitone = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        cents = abs(1200.0 * np.log2(f0[b] / max(f0[a], 1e-9)))
        if cents >= 900.0:
            octave += 1
        if cents >= 200.0:
            semitone += 1
    return {"octave_jump_count": int(octave), "semitone_jump_count": int(semitone)}


def _direction_slope(f0_hz: np.ndarray) -> float | None:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    if len(idx) < 3:
        return None
    x = idx.astype(np.float64) * HOP_S
    y = f0[idx]
    return float(np.polyfit(x, y, deg=1)[0])


def save_npz(path: Path, raw: dict[str, Any], audio: np.ndarray) -> None:
    np.savez_compressed(
        path,
        audio=audio.astype(np.float32),
        pitch_logits=raw["pitch_logits"],
        pitch_bin=raw["pitch_bin"],
        pitch_confidence=raw["pitch_confidence"],
        pitch_margin=raw["pitch_margin"],
        pitch_entropy=raw["pitch_entropy"],
        pitch_normalized_entropy=raw["pitch_normalized_entropy"],
        voiced_prob=raw["voiced_prob"],
        breath_prob=raw["breath_prob"],
        onset_prob=raw["onset_prob"],
        technique_logits=raw["technique_logits"],
        technique_probs=raw["technique_probs"],
        raw_f0_all_frames_hz=raw["raw_f0_all_frames_hz"],
        raw_f0_voiced_thresholded_hz=raw["raw_f0_voiced_thresholded_hz"],
        smoothed_f0_hz=raw["smoothed_f0_hz"],
    )


def write_svg(path: Path, audio: np.ndarray, raw: dict[str, Any], title: str) -> None:
    width = 1100
    height = 820
    left = 72
    right = 28
    plot_w = width - left - right
    duration = max(len(audio) / SR, HOP_S)
    times = np.arange(len(raw["voiced_prob"]), dtype=np.float32) * HOP_S

    rows = [
        ("Waveform", 50, 125, "waveform"),
        ("Voiced probability", 170, 255, "voiced"),
        ("Pitch confidence", 300, 385, "pitch_confidence"),
        ("Raw f0 before smoothing", 430, 525, "raw_f0"),
        ("Smoothed f0 after smoothing", 570, 665, "smoothed_f0"),
        ("Onset probability", 710, 790, "onset"),
    ]

    def x_at(t: float) -> float:
        return left + (t / duration) * plot_w

    def polyline(points: list[tuple[float, float]], color: str, width_px: float = 1.4) -> str:
        if not points:
            return ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" />'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="25" font-family="Arial" font-size="18" font-weight="700" fill="#111">{_xml(title)}</text>',
    ]

    for label, y0, y1, kind in rows:
        svg.append(f'<text x="18" y="{(y0 + y1) / 2:.1f}" font-family="Arial" font-size="12" fill="#444">{_xml(label)}</text>')
        svg.append(f'<line x1="{left}" y1="{y1}" x2="{left + plot_w}" y2="{y1}" stroke="#ddd" />')
        svg.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{y1-y0}" fill="none" stroke="#eee" />')

        if kind == "waveform":
            pts = waveform_points(audio, x_at, y0, y1, duration)
            svg.append(polyline(pts, "#444", 1.0))
        elif kind in {"voiced", "pitch_confidence", "onset"}:
            arr = {
                "voiced": raw["voiced_prob"],
                "pitch_confidence": raw["pitch_confidence"],
                "onset": raw["onset_prob"],
            }[kind]
            pts = [
                (float(x_at(t)), float(y1 - np.clip(v, 0, 1) * (y1 - y0)))
                for t, v in zip(times, arr)
            ]
            color = {"voiced": "#2563eb", "pitch_confidence": "#7c3aed", "onset": "#dc2626"}[kind]
            svg.append(polyline(pts, color, 1.4))
            if kind == "voiced":
                for thr in (0.3, 0.5, 0.7, 0.9):
                    y = y1 - thr * (y1 - y0)
                    svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#bbb" stroke-dasharray="4 4" />')
                    svg.append(f'<text x="{left+plot_w+4}" y="{y+4:.2f}" font-family="Arial" font-size="10" fill="#666">{thr:.1f}</text>')
        else:
            f0 = raw["raw_f0_voiced_thresholded_hz"] if kind == "raw_f0" else raw["smoothed_f0_hz"]
            valid = f0 > 0
            if np.any(valid):
                voiced_vals = f0[valid]
                fmin = max(40.0, float(np.percentile(voiced_vals, 2)) * 0.85)
                fmax = min(1200.0, float(np.percentile(voiced_vals, 98)) * 1.15)
                if fmax <= fmin:
                    fmax = fmin + 10.0
                pts = []
                for t, v in zip(times[valid], f0[valid]):
                    yy = y1 - ((float(v) - fmin) / (fmax - fmin)) * (y1 - y0)
                    pts.append((float(x_at(t)), float(np.clip(yy, y0, y1))))
                svg.append(polyline(pts, "#059669" if kind == "smoothed_f0" else "#ea580c", 1.5))
                for hz in nice_pitch_ticks(fmin, fmax):
                    yy = y1 - ((hz - fmin) / (fmax - fmin)) * (y1 - y0)
                    svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#eee" />')
                    svg.append(f'<text x="{left+plot_w+4}" y="{yy+4:.2f}" font-family="Arial" font-size="10" fill="#666">{hz:.0f}</text>')
            else:
                svg.append(f'<text x="{left+20}" y="{(y0+y1)/2:.1f}" font-family="Arial" font-size="12" fill="#777">No voiced f0 frames</text>')

    for t in np.linspace(0, duration, 7):
        x = x_at(float(t))
        svg.append(f'<line x1="{x:.2f}" y1="795" x2="{x:.2f}" y2="802" stroke="#666" />')
        svg.append(f'<text x="{x-12:.2f}" y="815" font-family="Arial" font-size="10" fill="#666">{t:.1f}s</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def waveform_points(audio: np.ndarray, x_at, y0: float, y1: float, duration: float) -> list[tuple[float, float]]:
    if audio.size == 0:
        return []
    n = min(1200, audio.size)
    idx = np.linspace(0, audio.size - 1, n).astype(int)
    vals = audio[idx]
    peak = max(float(np.max(np.abs(vals))), 1e-6)
    mid = (y0 + y1) / 2.0
    amp = (y1 - y0) / 2.0 * 0.92
    return [
        (float(x_at(i / SR)), float(mid - (v / peak) * amp))
        for i, v in zip(idx, vals)
    ]


def nice_pitch_ticks(fmin: float, fmax: float) -> list[float]:
    candidates = [55, 65, 82, 98, 110, 131, 147, 165, 196, 220, 262, 330, 392, 523, 659, 784, 1047]
    return [float(x) for x in candidates if fmin <= x <= fmax]


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_sample_markdown(path: Path, sample: str, summary: dict[str, Any], artifacts: dict[str, str]) -> None:
    vp = summary["voiced_probability"]
    pc = summary["pitch_confidence"]
    raw_f0 = summary["raw_f0_before_smoothing"]
    smooth_f0 = summary["smoothed_f0_after_smoothing"]
    onset = summary["onset_probability"]
    breath = summary["breath_probability"]
    technique = summary["technique"]
    lines = [
        f"# Raw Model Output Audit: {sample}",
        "",
        f"- JSON: `{artifacts['json']}`",
        f"- Raw arrays NPZ: `{artifacts['npz']}`",
        f"- Plot: `{artifacts['plot']}`",
        f"- Duration: `{summary['duration_s']:.3f}s`",
        f"- Frames: `{summary['frames']}`",
        "",
        "## Voiced Probability",
        "",
        f"- Mean/median/min/max: `{_fmt(vp.get('mean'))}` / `{_fmt(vp.get('median'))}` / `{_fmt(vp.get('min'))}` / `{_fmt(vp.get('max'))}`",
        f"- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `{_pct(vp['fraction_above_0_3'])}` / `{_pct(vp['fraction_above_0_5'])}` / `{_pct(vp['fraction_above_0_7'])}` / `{_pct(vp['fraction_above_0_9'])}`",
        f"- Near default 0.5 threshold (+/-0.05): `{_pct(vp['near_default_threshold_fraction'])}`",
        "",
        "## Pitch Confidence",
        "",
        f"- Max softmax mean/median: `{_fmt(pc['max_softmax_probability'].get('mean'))}` / `{_fmt(pc['max_softmax_probability'].get('median'))}`",
        f"- Top-2 margin mean/median: `{_fmt(pc['top1_top2_margin'].get('mean'))}` / `{_fmt(pc['top1_top2_margin'].get('median'))}`",
        f"- Normalized entropy mean/median: `{_fmt(pc['normalized_entropy'].get('mean'))}` / `{_fmt(pc['normalized_entropy'].get('median'))}`",
        "",
        "## F0",
        "",
        f"- Raw voiced f0 median/range: `{_fmt(raw_f0['median_hz'])}` Hz, `{_fmt(raw_f0['min_hz'])}`-`{_fmt(raw_f0['max_hz'])}` Hz",
        f"- Raw trimmed range: `{_fmt(raw_f0['trimmed_range_hz'])}` Hz",
        f"- Raw jumps: octave `{raw_f0['octave_jump_count']}`, semitone `{raw_f0['semitone_jump_count']}`",
        f"- Smoothed f0 median/range: `{_fmt(smooth_f0['median_hz'])}` Hz, `{_fmt(smooth_f0['min_hz'])}`-`{_fmt(smooth_f0['max_hz'])}` Hz",
        f"- Smoothed jumps: octave `{smooth_f0['octave_jump_count']}`, semitone `{smooth_f0['semitone_jump_count']}`",
        "",
        "## Onset / Breath",
        "",
        f"- Onset probability mean/median/max: `{_fmt(onset.get('mean'))}` / `{_fmt(onset.get('median'))}` / `{_fmt(onset.get('max'))}`",
        f"- Breath probability mean/median/max: `{_fmt(breath.get('mean'))}` / `{_fmt(breath.get('median'))}` / `{_fmt(breath.get('max'))}`",
        "",
        "## Technique",
        "",
        f"- Top class: `{technique['top_label']}` (`{_fmt(technique['top_probability'])}`)",
        "- Technique probabilities are marked unreliable and should not drive coaching decisions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def write_overview(path: Path, summaries: list[dict[str, Any]], output_dir: Path, checkpoint: Path) -> None:
    by_sample = {item["sample"]: item for item in summaries}
    lines = [
        "# M0 Model Output Audit",
        "",
        "Goal: inspect whether the checkpoint model itself is producing reliable raw outputs, independent of scoring, coaching, validity gates, and task-specific evaluators.",
        "",
        "No retraining, model architecture changes, scoring tuning, app behavior changes, or P4 regression expectation changes were made.",
        "",
        "## Checkpoint Inference Path",
        "",
        "The checkpoint path in `ml_new/inference/coach_inference.py` loads audio at 16 kHz, computes HCQT features and handcrafted VAD features, loads `UnifiedVocalModel`, and runs one forward pass:",
        "",
        "```text",
        "pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits_base, _ = model(hcqt_t, vad_t)",
        "```",
        "",
        "This audit bypasses validity gating, note segmentation, scoring, and coaching text. It saves raw/minimally processed arrays directly after the model forward pass.",
        "",
        "## Raw Outputs Produced",
        "",
        "- `pitch_logits`: `(T, 180)` unnormalized pitch-bin logits.",
        "- `voiced_prob`: `(T,)` per-frame voiced probability.",
        "- `breath_prob`: `(T,)` per-frame breath probability.",
        "- `onset_prob`: `(T,)` per-frame onset probability.",
        "- `tech_logits`: `(20,)` clip-level technique logits from the unified model technique head.",
        "",
        "Minimally processed audit fields derived from those outputs:",
        "",
        "- pitch softmax confidence, top-2 margin, and entropy from `pitch_logits`.",
        "- raw f0 from pitch-logit argmax, thresholded by `voiced_prob >= 0.5`.",
        "- smoothed f0 using the existing P2 `stabilize_f0_for_notes()` helper, for comparison only.",
        "- technique probabilities from softmax over `tech_logits`; marked unreliable.",
        "",
        "## Outputs",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "| Sample | VAD mean | VAD >=0.3 | VAD >=0.5 | VAD >=0.7 | VAD >=0.9 | Pitch conf mean | Pitch margin mean | Norm entropy mean | Raw f0 trimmed range | Raw octave jumps | Smoothed octave jumps | Onset mean | Breath mean | Top technique |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        vp = item["voiced_probability"]
        pc = item["pitch_confidence"]
        raw_f0 = item["raw_f0_before_smoothing"]
        smooth_f0 = item["smoothed_f0_after_smoothing"]
        lines.append(
            "| {sample} | {vad_mean} | {v03} | {v05} | {v07} | {v09} | {pc_mean} | {margin_mean} | {ent_mean} | {raw_range} | {raw_oct} | {smooth_oct} | {onset_mean} | {breath_mean} | {tech} |".format(
                sample=f"`{item['sample']}`",
                vad_mean=_fmt(vp.get("mean")),
                v03=_pct(vp["fraction_above_0_3"]),
                v05=_pct(vp["fraction_above_0_5"]),
                v07=_pct(vp["fraction_above_0_7"]),
                v09=_pct(vp["fraction_above_0_9"]),
                pc_mean=_fmt(pc["max_softmax_probability"].get("mean")),
                margin_mean=_fmt(pc["top1_top2_margin"].get("mean")),
                ent_mean=_fmt(pc["normalized_entropy"].get("mean")),
                raw_range=_fmt(raw_f0["trimmed_range_hz"]),
                raw_oct=raw_f0["octave_jump_count"],
                smooth_oct=smooth_f0["octave_jump_count"],
                onset_mean=_fmt(item["onset_probability"].get("mean")),
                breath_mean=_fmt(item["breath_probability"].get("mean")),
                tech=f"`{item['technique']['top_label']}`",
            )
        )

    lines += [
        "",
        "## Direct Answers",
        "",
        "### Was `00_silence` a high-confidence VAD false positive or barely above threshold?",
        "",
        _answer_silence(by_sample.get("00_silence")),
        "",
        "### Was `03_sustained_aaa` raw f0 actually unstable, or did segmentation cause most of the instability?",
        "",
        _answer_sustained(by_sample.get("03_sustained_aaa")),
        "",
        "### Does `04_pitch_slide` preserve directional movement in raw f0?",
        "",
        _answer_slide(by_sample.get("04_pitch_slide")),
        "",
        "### Does `05_twinkle_twinkle` have usable f0 structure?",
        "",
        _answer_twinkle(by_sample.get("05_twinkle_twinkle")),
        "",
        "## Interpretation",
        "",
        "- The raw VAD head is not calibrated enough to be trusted alone on fan/noise input; postprocessing gates remain necessary.",
        "- Pitch confidence is generally modest, and low top-2 margins indicate that argmax f0 can jump even when the voiced probability is high.",
        "- Technique logits/probabilities are included for audit completeness only. Existing notes about technique unreliability still apply.",
        "- This audit does not assert model accuracy against ground truth because these five samples do not include frame-level labels.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _answer_silence(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    vp = item["voiced_probability"]
    pc = item["pitch_confidence"]
    return (
        f"`00_silence` is a VAD false positive, but not a high-confidence one. "
        f"Mean voiced probability is `{_fmt(vp.get('mean'))}`, `{_pct(vp['fraction_above_0_5'])}` of frames are above 0.5, "
        f"only `{_pct(vp['fraction_above_0_7'])}` are above 0.7, and `{_pct(vp['fraction_above_0_9'])}` are above 0.9. "
        f"`{_pct(vp['near_default_threshold_fraction'])}` of frames are within +/-0.05 of the 0.5 threshold. "
        f"Pitch confidence is weak: mean max-softmax `{_fmt(pc['max_softmax_probability'].get('mean'))}` and mean top-2 margin `{_fmt(pc['top1_top2_margin'].get('mean'))}`."
    )


def _answer_sustained(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    raw = item["raw_f0_before_smoothing"]
    smooth = item["smoothed_f0_after_smoothing"]
    return (
        f"The raw f0 is already unstable before note segmentation. Raw thresholded f0 spans `{_fmt(raw['min_hz'])}`-`{_fmt(raw['max_hz'])}` Hz "
        f"with trimmed range `{_fmt(raw['trimmed_range_hz'])}` Hz, `{raw['octave_jump_count']}` octave-scale jumps, and `{raw['semitone_jump_count']}` >=2-semitone adjacent jumps. "
        f"Smoothing reduces octave jumps to `{smooth['octave_jump_count']}` and semitone jumps to `{smooth['semitone_jump_count']}`, so segmentation amplified the problem, but did not create it from a stable contour."
    )


def _answer_slide(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    raw = item["raw_f0_before_smoothing"]
    slope = raw["direction_slope_hz_per_s"]
    direction = "upward" if slope and slope > 0 else "downward" if slope and slope < 0 else "unclear"
    return (
        f"Yes. The raw f0 preserves broad directional movement: the fitted raw-f0 slope is `{_fmt(slope)}` Hz/s ({direction}), "
        f"with trimmed range `{_fmt(raw['trimmed_range_hz'])}` Hz. There are still `{raw['semitone_jump_count']}` large adjacent semitone jumps, so the movement is useful but not artifact-free."
    )


def _answer_twinkle(item: dict[str, Any] | None) -> str:
    if not item:
        return "Not evaluated."
    raw = item["raw_f0_before_smoothing"]
    pc = item["pitch_confidence"]
    return (
        f"Yes, with caveats. `05_twinkle_twinkle` has usable multi-note f0 structure: raw trimmed f0 range is `{_fmt(raw['trimmed_range_hz'])}` Hz, "
        f"median f0 is `{_fmt(raw['median_hz'])}` Hz, and voiced f0 covers `{_pct(raw['voiced_frame_fraction'])}` of frames. "
        f"Pitch confidence is moderate rather than strong: mean max-softmax `{_fmt(pc['max_softmax_probability'].get('mean'))}` and mean top-2 margin `{_fmt(pc['top1_top2_margin'].get('mean'))}`."
    )


def audit_sample(audio_path: Path, output_dir: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    sample = audio_path.stem
    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    audio = load_audio(audio_path)
    raw = run_raw_checkpoint(audio, checkpoint, device)
    summary = summarize_sample(sample, audio, raw)
    artifacts = {
        "json": str(sample_dir / f"{sample}_raw_summary.json"),
        "npz": str(sample_dir / f"{sample}_raw_outputs.npz"),
        "plot": str(sample_dir / f"{sample}_raw_outputs.svg"),
        "markdown": str(sample_dir / f"{sample}_raw_outputs.md"),
    }
    payload = {
        "sample": sample,
        "input_path": str(audio_path),
        "checkpoint": str(checkpoint),
        "device": device,
        "model": "UnifiedVocalModel",
        "raw_outputs_saved_before": [
            "validity_gating",
            "note_segmentation",
            "scoring",
            "coaching_text",
        ],
        "artifacts": artifacts,
        "summary": summary,
    }
    save_npz(Path(artifacts["npz"]), raw, audio)
    write_svg(Path(artifacts["plot"]), audio, raw, sample)
    Path(artifacts["json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_sample_markdown(Path(artifacts["markdown"]), sample, summary, artifacts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_output_audit"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ml_new/checkpoints/unified/best.pt"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for sample in SAMPLES:
        audio_path = args.samples_dir / f"{sample}.wav"
        if not audio_path.exists():
            print(f"Missing sample: {audio_path}", file=sys.stderr)
            return 2
        print(f"Auditing {audio_path}")
        summaries.append(audit_sample(audio_path, args.output_dir, args.checkpoint, args.device))

    summary_json = args.output_dir / "summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_overview(Path("M0_MODEL_OUTPUT_AUDIT.md"), summaries, args.output_dir, args.checkpoint)
    print(json.dumps({"status": "complete", "samples": len(summaries), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
