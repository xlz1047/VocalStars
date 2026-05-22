"""NanoPitchDataset: loads pre-extracted mel/f0/vad from NanoPitch .npz files.

Ported from NanoPitch training/train.py.  Serves (mel_clean, mel_noise, vad, f0)
windows for co-training VoiceCoachModel alongside singing datasets.

NPZ structure expected in data_dir:
  clean.npz — keys: mel (N,40) float16, f0 (N,) float16, vad (N,) float16,
                           lengths (C,) int32
  noise.npz — keys: mel (N,40) float16, lengths (C,) int32

The noise.npz is optional; if absent __getitem__ returns zeros for mel_noise.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Pitch conversion utilities (ported verbatim from NanoPitch model.py) ─────

_PITCH_FMIN: float = 31.7
_PITCH_CENTS_PER_BIN: float = 20.0
_PITCH_BINS: int = 360


def f0_to_posteriorgram(
    f0_hz: np.ndarray,
    n_frames: int | None = None,
    sigma_bins: float = 1.2,
) -> np.ndarray:
    """Create a Gaussian-blurred pitch posteriorgram from f0 values.

    For each voiced frame, places a Gaussian bump centred at the true pitch bin.
    This soft label helps the model learn because pitch is continuous, not discrete.

    Args:
        f0_hz: ``(T,)`` array of f0 in Hz (0 = unvoiced).
        n_frames: Number of output frames. Defaults to ``len(f0_hz)``.
        sigma_bins: Width of the Gaussian in bins (1.2 ≈ 24 cents).

    Returns:
        ``(T, 360)`` float32 array — one probability distribution per frame.
    """
    if n_frames is None:
        n_frames = len(f0_hz)

    f0_hz = np.asarray(f0_hz[:n_frames], dtype=np.float64)
    bins = _f0_to_bin(f0_hz)

    posteriorgram = np.zeros((n_frames, _PITCH_BINS), dtype=np.float32)
    bin_indices = np.arange(_PITCH_BINS, dtype=np.float64)

    for t in range(n_frames):
        if bins[t] < 0:
            continue
        dist = bin_indices - bins[t]
        posteriorgram[t] = np.exp(-0.5 * (dist / sigma_bins) ** 2)

    return posteriorgram


def _f0_to_bin(f0_hz: np.ndarray) -> np.ndarray:
    """Convert fundamental frequency (Hz) to pitch bin index.

    Returns -1 for unvoiced frames (f0 <= 0).
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    result = np.full_like(f0_hz, -1.0)
    voiced = f0_hz > 0
    result[voiced] = (
        1200.0 * np.log2(f0_hz[voiced] / _PITCH_FMIN) / _PITCH_CENTS_PER_BIN
    )
    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class NanoPitchDataset(Dataset):
    """PyTorch Dataset that serves (mel_clean, mel_noise, vad, f0) windows.

    Loads pre-extracted mel spectrograms from NanoPitch-format .npz files.
    Noise is sampled independently from clean data to allow batch-level mixing
    in the training loop.

    Args:
        data_dir: Directory containing ``clean.npz`` (and optionally ``noise.npz``).
        seq_len: Number of frames per training window (default 200 = 2 s at 10 ms hop).
    """

    def __init__(self, data_dir: str, seq_len: int = 200) -> None:
        self.seq_len = seq_len

        print("Loading clean.npz…")
        clean = np.load(os.path.join(data_dir, "clean.npz"))
        self.clean_mel: np.ndarray = clean["mel"]         # (total_frames, 40) float16
        self.clean_f0: np.ndarray = clean["f0"]           # (total_frames,)    float16
        self.clean_vad: np.ndarray = clean["vad"]         # (total_frames,)    float16
        self.clean_lengths: np.ndarray = clean["lengths"] # (num_clips,)       int32

        noise_path = os.path.join(data_dir, "noise.npz")
        self._has_noise = os.path.exists(noise_path)
        if self._has_noise:
            print("Loading noise.npz…")
            noise = np.load(noise_path)
            self.noise_mel: np.ndarray = noise["mel"]
            self.noise_lengths: np.ndarray = noise["lengths"]
            self.noise_segments = self._build_segments(self.noise_lengths, seq_len)
        else:
            self.noise_mel = np.zeros((1, 40), dtype=np.float16)
            self.noise_lengths = np.array([1], dtype=np.int32)
            self.noise_segments = [(0, 1)]

        self.clean_segments = self._build_segments(self.clean_lengths, seq_len)

        print(f"  Clean: {len(self.clean_mel):,} frames, "
              f"{len(self.clean_segments)} usable segments")
        if self._has_noise:
            print(f"  Noise: {len(self.noise_mel):,} frames, "
                  f"{len(self.noise_segments)} usable segments")

        self.rng = np.random.default_rng()

    def _build_segments(
        self, lengths: np.ndarray, min_len: int
    ) -> list[tuple[int, int]]:
        """Return (start, end) index pairs for clips that are at least min_len frames."""
        segments: list[tuple[int, int]] = []
        offset = 0
        for length in lengths:
            if length >= min_len:
                segments.append((offset, int(offset + length)))
            offset += int(length)
        return segments

    def __len__(self) -> int:
        """Return a reasonable epoch size (capped to avoid very long CPUs epochs)."""
        return min(len(self.clean_segments) * 3, 10000)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random window from clean data and a random noise window.

        Returns:
            Tuple of float32 tensors:
                mel_clean: ``(seq_len, 40)``
                mel_noise: ``(seq_len, 40)``
                vad:       ``(seq_len,)``
                f0:        ``(seq_len,)``
        """
        seg_idx = int(self.rng.integers(len(self.clean_segments)))
        start, end = self.clean_segments[seg_idx]
        offset = int(self.rng.integers(0, end - start - self.seq_len + 1))
        s = start + offset

        mel_clean = self.clean_mel[s : s + self.seq_len].astype(np.float32)
        f0 = self.clean_f0[s : s + self.seq_len].astype(np.float32)
        vad = self.clean_vad[s : s + self.seq_len].astype(np.float32)

        if self._has_noise and len(self.noise_segments) > 0:
            ni = int(self.rng.integers(len(self.noise_segments)))
            ns, ne = self.noise_segments[ni]
            n_off = int(self.rng.integers(0, ne - ns - self.seq_len + 1))
            mel_noise = self.noise_mel[ns + n_off : ns + n_off + self.seq_len].astype(np.float32)
        else:
            mel_noise = np.zeros_like(mel_clean)

        return (
            torch.from_numpy(mel_clean),
            torch.from_numpy(mel_noise),
            torch.from_numpy(vad),
            torch.from_numpy(f0),
        )
