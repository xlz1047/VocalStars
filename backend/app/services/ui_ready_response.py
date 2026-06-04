"""Build UI-ready analysis responses for the audio ML endpoint."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_new.inference.coach_inference import CoachingResult, HOP_LENGTH, HOP_S, SR  # noqa: E402
from scripts.eval import ui_ready_analysis as h2  # noqa: E402
from scripts.eval.coaching_categories import build_coaching_categories  # noqa: E402
from scripts.eval.evaluate_task_specific import apply_h3  # noqa: E402


def _safe_float(value: Any, precision: int = 6) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return round(val, precision)


def _load_audio(audio_path: Path) -> np.ndarray:
    from ml_new.inference.coach_inference import _load_audio_robust

    return _load_audio_robust(audio_path, sr=SR)


def _decimate_time_matrix(matrix: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    """Return time x feature matrix with bounded frame count."""
    if matrix.shape[0] <= max_frames:
        return matrix, 1
    stride = int(np.ceil(matrix.shape[0] / max_frames))
    groups = []
    for start in range(0, matrix.shape[0], stride):
        groups.append(np.mean(matrix[start : start + stride], axis=0))
    return np.asarray(groups, dtype=np.float32), stride


def _build_log_mel_spectrogram(audio_path: Path, max_frames: int = 220, n_mels: int = 48) -> dict[str, Any] | None:
    """Build a compact log-mel spectrogram for UI visualization.

    This is display data, not a model input contract. Values are normalized to
    0..1 over an 80 dB display range so the payload stays small and safe to
    render directly.
    """
    try:
        import librosa

        audio = _load_audio(audio_path)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=512,
            hop_length=HOP_LENGTH,
            win_length=400,
            n_mels=n_mels,
            fmin=50.0,
            fmax=SR / 2.0,
            power=2.0,
            htk=True,
        )
        db = librosa.power_to_db(mel, ref=np.max)
        normalized = np.clip((db + 80.0) / 80.0, 0.0, 1.0).T.astype(np.float32)
        compact, stride = _decimate_time_matrix(normalized, max_frames=max_frames)
        frame_times = [round(i * HOP_S * stride, 6) for i in range(compact.shape[0])]
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=50.0, fmax=SR / 2.0, htk=True)
        return {
            "kind": "log_mel_spectrogram",
            "is_display_downsampled": True,
            "sample_rate": SR,
            "hop_s": round(HOP_S * stride, 6),
            "source_hop_s": HOP_S,
            "frame_stride": stride,
            "n_mels": n_mels,
            "frequency_min_hz": 50.0,
            "frequency_max_hz": SR / 2.0,
            "mel_scale": "htk",
            "value_scale": "normalized_db_80db_range",
            "time_s": frame_times,
            "mel_frequencies_hz": [_safe_float(freq, 3) for freq in mel_freqs],
            "values": [[_safe_float(value, 4) for value in row] for row in compact],
            "caveat": "Display spectrogram only; do not treat it as timbre, breath support, strain, or vocal health diagnosis.",
        }
    except Exception as exc:
        return {
            "kind": "log_mel_spectrogram",
            "error": str(exc),
            "caveat": "Spectrogram visualization could not be generated for this response.",
        }


def _signal_quality_risk(frame: dict[str, Any]) -> float:
    quality = frame.get("signal_quality") or {}
    risk = 0.0
    if quality.get("near_silence"):
        risk += 0.35
    if quality.get("noisy"):
        risk += 0.3
    if quality.get("low_confidence"):
        risk += 0.25
    if quality.get("source_disagreement"):
        risk += 0.2
    if quality.get("clipped"):
        risk += 0.35
    return float(max(0.0, min(1.0, risk)))


def _build_posteriorgram(frames: list[dict[str, Any]], max_frames: int = 220) -> dict[str, Any]:
    rows = []
    for frame in frames:
        rows.append(
            [
                float(frame.get("voice_confidence") or 0.0),
                float(frame.get("pitch_confidence") or 0.0),
                _signal_quality_risk(frame),
            ]
        )
    matrix = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    compact, stride = _decimate_time_matrix(matrix, max_frames=max_frames)
    return {
        "kind": "posterior_confidence_summary",
        "is_display_downsampled": True,
        "hop_s": round(HOP_S * stride, 6),
        "source_hop_s": HOP_S,
        "frame_stride": stride,
        "row_labels": ["voice_confidence", "pitch_confidence", "signal_quality_risk"],
        "time_s": [round(i * HOP_S * stride, 6) for i in range(compact.shape[0])],
        "values": [[_safe_float(value, 4) for value in row] for row in compact],
        "caveat": "Posterior/confidence summary for visualization; confidence values are source-dependent and not absolute vocal quality scores.",
    }


def _build_visualizations(audio_path: Path, frames: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "ui_visualizations.v1",
        "spectrogram": _build_log_mel_spectrogram(audio_path),
        "posteriorgram": _build_posteriorgram(frames),
    }


def build_ui_ready_response(
    audio_path: str | Path,
    *,
    coaching_result: CoachingResult,
    task_config: dict[str, Any] | None,
    checkpoint: Path,
    device: str = "cpu",
    include_frames: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Return H2/H3/H4-style UI-ready analysis for one uploaded file."""
    total_start = time.perf_counter()
    effective_task_config = task_config or coaching_result.task_config or {"task_type": "free_singing"}

    h2_start = time.perf_counter()
    analysis = h2.build_ui_ready_analysis(
        Path(audio_path),
        checkpoint=checkpoint,
        device=device,
        debug=debug,
        task_config=effective_task_config,
    )
    h2_elapsed = time.perf_counter() - h2_start

    # The H2 builder was originally report-oriented and may infer validity from
    # historical eval files. For live uploads, the checkpoint inference result is
    # the source of truth for validity and task semantics.
    analysis["analysis_validity"] = coaching_result.analysis_validity
    analysis["task_config"] = effective_task_config
    task_start = time.perf_counter()
    analysis = apply_h3(analysis, effective_task_config)
    task_elapsed = time.perf_counter() - task_start

    source_strategy = (analysis.get("debug") or {}).get("source_strategy")
    if source_strategy is not None:
        analysis["source_strategy"] = source_strategy
    analysis["caveats"] = analysis.get("feedback_policy", {}).get("caveats", [])
    analysis["coaching_categories"] = build_coaching_categories(analysis)

    if not include_frames:
        analysis["frames"] = []
        analysis["visualizations"] = {
            "schema_version": "ui_visualizations.v1",
            "spectrogram": None,
            "posteriorgram": None,
        }
        visualization_elapsed = 0.0
    else:
        vis_start = time.perf_counter()
        analysis["visualizations"] = _build_visualizations(Path(audio_path), analysis.get("frames") or [])
        visualization_elapsed = time.perf_counter() - vis_start

    audio_duration = None
    audio_meta = analysis.get("audio") or {}
    if isinstance(audio_meta, dict):
        audio_duration = audio_meta.get("duration_s")
    total_elapsed = time.perf_counter() - total_start
    realtime_factor = None
    if isinstance(audio_duration, (int, float)) and audio_duration > 0:
        realtime_factor = total_elapsed / float(audio_duration)

    analysis["performance"] = {
        "schema_version": "ui_ready_performance.v1",
        "device": device,
        "include_frames": include_frames,
        "debug": debug,
        "audio_duration_s": _safe_float(audio_duration),
        "h2_frame_export_s": _safe_float(h2_elapsed, 4),
        "task_evaluator_s": _safe_float(task_elapsed, 4),
        "visualization_s": _safe_float(visualization_elapsed, 4),
        "ui_ready_total_s": _safe_float(total_elapsed, 4),
        "ui_ready_realtime_factor": _safe_float(realtime_factor, 4),
        "caveat": "Backend wall-clock timing for this request; browser, network, cold start, and hardware vary.",
    }

    return analysis
