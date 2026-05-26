"""Ground truth label extraction for pitch (F0) and VAD.

Two label types are produced:
  - f0_hz (T,): Per-frame fundamental frequency in Hz; 0.0 on unvoiced frames.
  - vad (T,): Per-frame binary voice activity; uint8 (0 or 1).

Label sources vary by dataset:
  - GTSinger: phoneme boundary JSON files provide accurate frame-level VAD.
  - VocalSet / PopBuTFy: VAD derived from RMS energy threshold.

For F0, two backends are available:
  - ``fast`` (default): librosa.yin — ~100x faster, sufficient for training.
  - ``accurate``: librosa.pyin — probabilistic, better for evaluation labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import librosa


def extract_f0(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 65.4,
    fmax: float = 2093.0,
    method: str = "fast",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame F0 from audio.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        hop_length: Frame shift in samples.
        fmin: Minimum detectable frequency in Hz (C2 = 65.4 Hz).
        fmax: Maximum detectable frequency in Hz (C7 = 2093.0 Hz).
        method: ``"fast"`` uses librosa.yin (~0.07 s/clip), ``"accurate"``
            uses librosa.pyin (~7.5 s/clip but returns voicing probabilities).

    Returns:
        Tuple ``(f0_hz, voiced_flag, voiced_probs)``, each shape ``(T,)``.
        ``f0_hz`` is 0.0 on unvoiced frames.
        ``voiced_flag`` is a boolean array.
        ``voiced_probs`` is per-frame voicing confidence in [0, 1]; all 1.0
        for the fast/yin path (yin has no confidence estimate).
    """
    if method == "accurate":
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
        )
        f0_hz = np.where(voiced_flag, f0, 0.0).astype(np.float32)
        return f0_hz, voiced_flag.astype(bool), voiced_probs.astype(np.float32)

    # Fast path: yin + energy-based voicing
    f0 = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop_length)[0]
    T = min(len(f0), len(rms))
    f0 = f0[:T].astype(np.float32)
    rms = rms[:T]
    # Voiced if energy is above noise floor and yin returned a pitch in range
    voiced_flag = (rms > 0.005) & (f0 >= fmin) & (f0 <= fmax)
    f0_hz = np.where(voiced_flag, f0, 0.0).astype(np.float32)
    return f0_hz, voiced_flag, np.ones(T, dtype=np.float32)


def extract_vad_gtsinger(
    json_path: str | Path,
    n_frames: int,
    sr: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """Build frame-level VAD labels from a GTSinger phoneme boundary JSON.

    A frame is voiced (1) if any phoneme interval overlaps it.

    Args:
        json_path: Path to the ``.json`` sidecar with ``ph_start``/``ph_end`` arrays.
        n_frames: Total number of frames in the clip.
        sr: Sample rate in Hz.
        hop_length: Frame shift in samples.

    Returns:
        uint8 array of shape ``(n_frames,)`` with values 0 or 1.
    """
    vad = np.zeros(n_frames, dtype=np.uint8)
    try:
        with open(json_path) as fh:
            entries = json.load(fh)
        for entry in entries:
            starts = entry.get("ph_start", [])
            ends = entry.get("ph_end", [])
            for t_start, t_end in zip(starts, ends):
                frame_start = int(t_start * sr / hop_length)
                frame_end = int(np.ceil(t_end * sr / hop_length))
                frame_start = max(0, frame_start)
                frame_end = min(n_frames, frame_end)
                if frame_end > frame_start:
                    vad[frame_start:frame_end] = 1
    except Exception:
        pass
    return vad


def extract_vad_energy(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    energy_threshold: float = 0.01,
) -> np.ndarray:
    """Build per-frame VAD labels from RMS energy alone.

    Used for datasets without phoneme annotations (VocalSet, PopBuTFy).

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        hop_length: Frame shift in samples.
        energy_threshold: Minimum RMS for a frame to be voiced.

    Returns:
        uint8 array of shape ``(T,)`` with values 0 or 1.
    """
    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop_length)[0]
    return (rms > energy_threshold).astype(np.uint8)
