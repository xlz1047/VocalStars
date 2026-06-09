#!/usr/bin/env python3
"""Decoupled F0 extraction protocol for the reference-track vectorisation pipeline.

The production inference stack (coach_inference.py, streaming_inference.py) is
never imported here. Swap extractors by passing --extractor at the CLI.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class PitchExtractor(ABC):
    """Extract fundamental frequency from a mono audio signal."""

    hop_s: float  # seconds per output frame (class-level constant)

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (f0_hz, voiced) arrays of equal length.

        Args:
            audio: Mono float32 signal, range [-1, 1].
            sr: Sample rate in Hz.

        Returns:
            f0_hz: float32 array of length T. 0.0 indicates unvoiced.
            voiced: bool array of length T.
        """


class PyinExtractor(PitchExtractor):
    """librosa.pyin — probabilistic YIN with voiced-probability weighting."""

    hop_s: float = 0.01  # 10 ms — matches model HOP_S
    fmin: float = 65.4   # C2
    fmax: float = 2093.0 # C7

    def __init__(self, fmin: float = 65.4, fmax: float = 2093.0) -> None:
        self.fmin = fmin
        self.fmax = fmax

    def extract(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        hop_length = max(1, int(round(self.hop_s * sr)))
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=hop_length,
            fill_na=0.0,
        )
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
        voiced = np.asarray(voiced_flag, dtype=bool)
        f0[~voiced] = 0.0
        return f0, voiced


class GroundTruthExtractor(PitchExtractor):
    """Parses MIR-1K .pv sidecar files — no inference required.

    Each .pv file is a plain-text sequence of F0 values (one per line) at a
    fixed 20 ms hop (50 Hz frame rate). 0.0 means unvoiced.
    """

    hop_s: float = 0.02  # 20 ms — MIR-1K annotation rate

    def __init__(self, pv_path: Path) -> None:
        self._pv_path = pv_path

    def extract(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        lines = self._pv_path.read_text(encoding="utf-8").splitlines()
        values: list[float] = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    values.append(0.0)
        f0 = np.asarray(values, dtype=np.float32)
        voiced = f0 > 0.0
        return f0, voiced


def resample_to_10ms(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    source_hop_s: float,
    target_hop_s: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an F0 contour to a canonical 10 ms hop via linear interpolation.

    Unvoiced frames are not interpolated — a frame is voiced only if both
    neighbours in the source grid are voiced. This avoids smearing pitch values
    across silence boundaries.
    """
    if abs(source_hop_s - target_hop_s) < 1e-6:
        return f0_hz.copy(), voiced.copy()

    n_src = len(f0_hz)
    duration_s = n_src * source_hop_s
    n_tgt = max(1, int(round(duration_s / target_hop_s)))

    src_times = np.arange(n_src, dtype=np.float64) * source_hop_s
    tgt_times = np.arange(n_tgt, dtype=np.float64) * target_hop_s

    # Interpolate only voiced segments; unvoiced stays 0
    voiced_f = voiced.astype(np.float32)
    f0_interp = np.interp(tgt_times, src_times, f0_hz)
    v_interp = np.interp(tgt_times, src_times, voiced_f) >= 0.5

    f0_out = np.where(v_interp, f0_interp, 0.0).astype(np.float32)
    return f0_out, v_interp


def make_extractor(name: str, **kwargs) -> PitchExtractor:
    """Factory for the --extractor CLI flag."""
    name = name.lower().strip()
    if name in {"pyin", "librosa"}:
        return PyinExtractor(**kwargs)
    raise ValueError(
        f"Unknown extractor '{name}'. Choices: pyin. "
        "For ground-truth extraction (MIR-1K), GroundTruthExtractor is instantiated "
        "automatically from the sidecar path."
    )


# ---------------------------------------------------------------------------
# Note name → Hz lookup (A4 = 440 Hz, MIDI standard)
# ---------------------------------------------------------------------------

_NOTE_RE = re.compile(r"^([A-Ga-g])(#|b)?(\d+)$")
_SEMITONE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_name_to_hz(note: str) -> float | None:
    """Convert a note name like 'C4' or 'A#3' to Hz. Returns None if unparsable."""
    m = _NOTE_RE.match(note.strip())
    if not m:
        return None
    letter, accidental, octave_str = m.group(1).upper(), m.group(2), m.group(3)
    semitone = _SEMITONE_MAP[letter]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = 12 * (int(octave_str) + 1) + semitone
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))
