#!/usr/bin/env python3
"""Run NanoPitch on an arbitrary WAV/audio file and export raw VAD/f0 outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from ml_new.nanopitch.model import NanoPitch, viterbi_decode, viterbi_decode_realtime  # noqa: E402


SR = 16_000
N_FFT = 512
N_FREQS = N_FFT // 2 + 1
WIN_LENGTH = 400
HOP_LENGTH = 160
HOP_S = HOP_LENGTH / SR
N_MELS = 40
LOG_OFFSET = 1e-10
VOICED_THRESHOLD = 0.5
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "nanopitch_best.pth"


def load_audio(path: Path) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(audio, dtype=np.float32)


def htk_hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def htk_mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank() -> np.ndarray:
    mel_min = float(htk_hz_to_mel(0.0))
    mel_max = float(htk_hz_to_mel(SR / 2.0))
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2, dtype=np.float64)
    hz_points = htk_mel_to_hz(mel_points)
    bin_points = hz_points * N_FFT / SR

    mel_fb = np.zeros((N_MELS, N_FREQS), dtype=np.float32)
    for m in range(N_MELS):
        left = float(bin_points[m])
        center = float(bin_points[m + 1])
        right = float(bin_points[m + 2])
        for k in range(N_FREQS):
            if left <= k < center:
                mel_fb[m, k] = (k - left) / max(center - left, 1e-12)
            elif center <= k <= right:
                mel_fb[m, k] = (right - k) / max(right - center, 1e-12)
    return mel_fb


def compute_nanopitch_logmel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel frames matching NanoPitch's C/WASM overlap-save path."""
    mel_fb = build_mel_filterbank()
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH))
    hann = hann.astype(np.float32)

    n_frames = int(np.ceil(len(audio) / HOP_LENGTH)) if len(audio) else 1
    padded = np.zeros(n_frames * HOP_LENGTH, dtype=np.float32)
    padded[: len(audio)] = audio

    analysis_mem = np.zeros(WIN_LENGTH, dtype=np.float32)
    frames = np.zeros((n_frames, N_MELS), dtype=np.float32)
    overlap = WIN_LENGTH - HOP_LENGTH

    for i in range(n_frames):
        hop = padded[i * HOP_LENGTH : (i + 1) * HOP_LENGTH]
        window_buf = np.empty(WIN_LENGTH, dtype=np.float32)
        window_buf[:overlap] = analysis_mem[HOP_LENGTH:]
        window_buf[overlap:] = hop
        analysis_mem[:] = window_buf

        windowed = window_buf * hann
        fft = np.fft.rfft(windowed, n=N_FFT)
        power = (fft.real * fft.real + fft.imag * fft.imag).astype(np.float32)
        mel_energy = mel_fb @ power
        frames[i] = np.log(mel_energy + LOG_OFFSET).astype(np.float32)
    return frames


def load_model(checkpoint: Path, device: str) -> NanoPitch:
    dev = torch.device(device)
    try:
        ckpt = torch.load(str(checkpoint), map_location=dev, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=dev)
    kwargs = ckpt.get("model_kwargs", {"cond_size": 64, "gru_size": 96})
    model = NanoPitch(**kwargs).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_nanopitch(audio: np.ndarray, checkpoint: Path, device: str) -> dict[str, Any]:
    start = time.perf_counter()
    mel = compute_nanopitch_logmel(audio)
    model = load_model(checkpoint, device)
    dev = torch.device(device)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(dev)

    with torch.no_grad():
        vad_t, pitch_t, _ = model(mel_t)

    vad_prob = vad_t.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)
    pitch_post = pitch_t.squeeze(0).cpu().numpy().astype(np.float32)
    f0_offline = viterbi_decode(pitch_post).astype(np.float32)
    f0_realtime = viterbi_decode_realtime(pitch_post).astype(np.float32)
    top2_idx = np.argsort(pitch_post, axis=1)[:, -2:]
    top1 = pitch_post[np.arange(len(pitch_post)), top2_idx[:, 1]]
    top2 = pitch_post[np.arange(len(pitch_post)), top2_idx[:, 0]]
    runtime_s = time.perf_counter() - start

    return {
        "mel": mel,
        "vad_prob": vad_prob,
        "pitch_posterior": pitch_post,
        "pitch_confidence": top1.astype(np.float32),
        "pitch_margin": (top1 - top2).astype(np.float32),
        "f0_hz": f0_offline,
        "f0_hz_offline": f0_offline,
        "f0_hz_realtime": f0_realtime,
        "runtime_s": float(runtime_s),
    }


def array_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "percentiles": {}}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {f"p{p:02d}": float(np.percentile(arr, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
    }


def f0_summary(f0_hz: np.ndarray) -> dict[str, Any]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    voiced = f0[valid]
    duration_s = max(float(len(f0) * HOP_S), HOP_S)
    jumps = jump_metrics(f0)
    if voiced.size == 0:
        return {
            "coverage": 0.0,
            "mean_hz": None,
            "median_hz": None,
            "min_hz": None,
            "max_hz": None,
            "trimmed_p05_hz": None,
            "trimmed_p95_hz": None,
            "trimmed_range_hz": None,
            **jumps,
            "octave_jump_rate_per_second": 0.0,
            "semitone_jump_rate_per_second": 0.0,
            "f0_stability_cents": None,
            "direction_slope_hz_per_s": None,
        }
    return {
        "coverage": float(np.mean(valid)),
        "mean_hz": float(np.mean(voiced)),
        "median_hz": float(np.median(voiced)),
        "min_hz": float(np.min(voiced)),
        "max_hz": float(np.max(voiced)),
        "trimmed_p05_hz": float(np.percentile(voiced, 5)),
        "trimmed_p95_hz": float(np.percentile(voiced, 95)),
        "trimmed_range_hz": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)),
        **jumps,
        "octave_jump_rate_per_second": float(jumps["octave_jump_count"] / duration_s),
        "semitone_jump_rate_per_second": float(jumps["semitone_jump_count"] / duration_s),
        "f0_stability_cents": f0_stability_cents(f0),
        "direction_slope_hz_per_s": direction_slope(f0),
    }


def jump_metrics(f0_hz: np.ndarray) -> dict[str, int]:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
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


def f0_stability_cents(f0_hz: np.ndarray) -> float | None:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size < 3:
        return None
    cents = 1200.0 * np.log2(voiced / max(float(np.median(voiced)), 1e-9))
    lo, hi = np.percentile(cents, [5, 95])
    trimmed = cents[(cents >= lo) & (cents <= hi)]
    if trimmed.size < 3:
        trimmed = cents
    return float(np.std(trimmed))


def direction_slope(f0_hz: np.ndarray) -> float | None:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    if idx.size < 3:
        return None
    x = idx.astype(np.float64) * HOP_S
    y = f0[idx]
    return float(np.polyfit(x, y, deg=1)[0])


def serialize_array(values: np.ndarray, precision: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return [round(float(x), precision) if np.isfinite(x) else None for x in arr]


def evaluate_audio(audio_path: Path, output_dir: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    sample = audio_path.stem
    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    audio = load_audio(audio_path)
    raw = run_nanopitch(audio, checkpoint, device)
    frame_times = np.arange(len(raw["vad_prob"]), dtype=np.float32) * HOP_S
    voiced_mask = raw["vad_prob"] >= VOICED_THRESHOLD

    json_path = sample_dir / f"{sample}_nanopitch.json"
    plot_path = sample_dir / f"{sample}_nanopitch.svg"

    payload = {
        "model_name": "NanoPitch",
        "input_path": str(audio_path),
        "checkpoint_path": str(checkpoint),
        "device": device,
        "sample_rate": SR,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "hop_s": HOP_S,
        "n_mels": N_MELS,
        "mel_scale": "HTK",
        "log_offset": LOG_OFFSET,
        "duration_s": float(len(audio) / SR),
        "n_frames": int(len(frame_times)),
        "runtime_s": raw["runtime_s"],
        "frame_times": serialize_array(frame_times),
        "vad_prob": serialize_array(raw["vad_prob"]),
        "voiced_mask": [bool(x) for x in voiced_mask],
        "f0_hz": serialize_array(raw["f0_hz"]),
        "f0_hz_offline": serialize_array(raw["f0_hz_offline"]),
        "f0_hz_realtime": serialize_array(raw["f0_hz_realtime"]),
        "pitch_confidence": serialize_array(raw["pitch_confidence"]),
        "pitch_margin": serialize_array(raw["pitch_margin"]),
        "f0_statistics": {
            "offline": f0_summary(raw["f0_hz_offline"]),
            "realtime": f0_summary(raw["f0_hz_realtime"]),
        },
        "vad_statistics": {
            **array_summary(raw["vad_prob"]),
            "threshold": VOICED_THRESHOLD,
            "voiced_frame_ratio": float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0,
            "fraction_above_0_3": float(np.mean(raw["vad_prob"] >= 0.3)) if len(raw["vad_prob"]) else 0.0,
            "fraction_above_0_5": float(np.mean(raw["vad_prob"] >= 0.5)) if len(raw["vad_prob"]) else 0.0,
            "fraction_above_0_7": float(np.mean(raw["vad_prob"] >= 0.7)) if len(raw["vad_prob"]) else 0.0,
            "fraction_above_0_9": float(np.mean(raw["vad_prob"] >= 0.9)) if len(raw["vad_prob"]) else 0.0,
        },
        "pitch_confidence_statistics": {
            "max_posterior": array_summary(raw["pitch_confidence"]),
            "top1_top2_margin": array_summary(raw["pitch_margin"]),
        },
        "artifacts": {
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_svg(plot_path, audio, frame_times, raw["vad_prob"], raw["f0_hz"], raw["pitch_confidence"], sample)
    return payload


def write_svg(
    path: Path,
    audio: np.ndarray,
    frame_times: np.ndarray,
    vad_prob: np.ndarray,
    f0_hz: np.ndarray,
    pitch_confidence: np.ndarray,
    title: str,
) -> None:
    width = 1100
    height = 650
    left = 76
    right = 30
    plot_w = width - left - right
    duration = max(float(len(audio) / SR), HOP_S)
    rows = [
        ("Waveform", 50, 130),
        ("VAD probability", 175, 270),
        ("F0", 320, 475),
        ("Pitch confidence", 520, 610),
    ]

    def x_at(t: float) -> float:
        return left + (t / duration) * plot_w

    def polyline(points: list[tuple[float, float]], color: str, width_px: float = 1.4, extra: str = "") -> str:
        if not points:
            return ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width_px}" {extra}/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="26" font-family="Arial" font-size="18" font-weight="700" fill="#111">NanoPitch WAV eval: {escape_xml(title)}</text>',
    ]
    for label, y0, y1 in rows:
        svg.append(f'<text x="16" y="{(y0 + y1) / 2:.1f}" font-family="Arial" font-size="12" fill="#444">{escape_xml(label)}</text>')
        svg.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{y1-y0}" fill="none" stroke="#eee" />')
        svg.append(f'<line x1="{left}" y1="{y1}" x2="{left+plot_w}" y2="{y1}" stroke="#ddd" />')

    svg.append(polyline(waveform_points(audio, x_at, 50, 130), "#444", 1.0))
    svg.append(polyline(unit_points(frame_times, vad_prob, x_at, 175, 270), "#2563eb", 1.4))
    for thr in (0.3, 0.5, 0.7, 0.9):
        y = 270 - thr * (270 - 175)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#bbb" stroke-dasharray="4 4" />')
        svg.append(f'<text x="{left+plot_w+4}" y="{y+4:.2f}" font-family="Arial" font-size="10" fill="#666">{thr:.1f}</text>')

    valid = f0_hz > 0
    if np.any(valid):
        vals = f0_hz[valid]
        fmin = max(30.0, float(np.percentile(vals, 2)) * 0.85)
        fmax = min(2200.0, float(np.percentile(vals, 98)) * 1.15)
        if fmax <= fmin:
            fmax = fmin + 10.0
        pts = []
        for t, hz in zip(frame_times[valid], f0_hz[valid]):
            yy = 475 - ((float(hz) - fmin) / (fmax - fmin)) * (475 - 320)
            pts.append((float(x_at(float(t))), float(np.clip(yy, 320, 475))))
        svg.append(polyline(pts, "#059669", 1.5))
        for hz in nice_pitch_ticks(fmin, fmax):
            yy = 475 - ((hz - fmin) / (fmax - fmin)) * (475 - 320)
            svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#eee" />')
            svg.append(f'<text x="{left+plot_w+4}" y="{yy+4:.2f}" font-family="Arial" font-size="10" fill="#666">{hz:.0f}</text>')
    else:
        svg.append(f'<text x="{left+20}" y="400" font-family="Arial" font-size="12" fill="#777">No decoded f0</text>')

    svg.append(polyline(unit_points(frame_times, pitch_confidence, x_at, 520, 610), "#7c3aed", 1.4))
    for t in np.linspace(0, duration, 7):
        x = x_at(float(t))
        svg.append(f'<line x1="{x:.2f}" y1="620" x2="{x:.2f}" y2="627" stroke="#666" />')
        svg.append(f'<text x="{x-12:.2f}" y="642" font-family="Arial" font-size="10" fill="#666">{t:.1f}s</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def waveform_points(audio: np.ndarray, x_at, y0: float, y1: float) -> list[tuple[float, float]]:
    if audio.size == 0:
        return []
    n = min(1400, audio.size)
    idx = np.linspace(0, audio.size - 1, n).astype(int)
    vals = audio[idx]
    peak = max(float(np.max(np.abs(vals))), 1e-6)
    mid = (y0 + y1) / 2.0
    amp = (y1 - y0) / 2.0 * 0.9
    return [(float(x_at(i / SR)), float(mid - (v / peak) * amp)) for i, v in zip(idx, vals)]


def unit_points(times: np.ndarray, values: np.ndarray, x_at, y0: float, y1: float) -> list[tuple[float, float]]:
    return [
        (float(x_at(float(t))), float(y1 - np.clip(float(v), 0.0, 1.0) * (y1 - y0)))
        for t, v in zip(times, values)
    ]


def nice_pitch_ticks(fmin: float, fmax: float) -> list[float]:
    candidates = [32, 55, 65, 82, 98, 110, 131, 147, 165, 196, 220, 262, 330, 392, 440, 523, 659, 784, 1047, 1568, 2093]
    return [float(x) for x in candidates if fmin <= x <= fmax]


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to WAV/audio file.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/nanopitch_eval"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.exists():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate_audio(args.audio, args.output_dir, args.checkpoint, args.device)
    print(json.dumps({"status": "complete", "json": payload["artifacts"]["json"], "plot": payload["artifacts"]["plot"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
