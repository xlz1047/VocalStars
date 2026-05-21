"""Visualization utilities for ML evaluation plots.

Generates matplotlib plots for pitch curves, energy envelopes, and beat alignment;
saves results to PNG files in a results directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_pitch_curve(f0: np.ndarray, times: np.ndarray, ground_truth: np.ndarray | None = None, save_path: str | Path | None = None):
    """Plot F0 contour (estimated and optionally ground truth)."""
    if not MATPLOTLIB_AVAILABLE:
        return

    save_path = Path(save_path) if save_path else RESULTS_DIR / "pitch_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    f0_clean = np.nan_to_num(f0, nan=0.0)
    ax.plot(times, f0_clean, label="Estimated F0", linewidth=1.5, color="blue")
    if ground_truth is not None:
        ax.plot(times, ground_truth, label="Ground Truth", linewidth=1.5, color="red", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Pitch Contour")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pitch curve plot to {save_path}")


def plot_energy_envelope(energy: np.ndarray, times: np.ndarray, breaths: list[float] | None = None, save_path: str | Path | None = None):
    """Plot energy (RMS) contour and optionally detected breath times."""
    if not MATPLOTLIB_AVAILABLE:
        return

    save_path = Path(save_path) if save_path else RESULTS_DIR / "energy_envelope.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, energy, label="RMS Energy", linewidth=1.5, color="green")
    if breaths:
        ax.vlines(breaths, 0, energy.max() * 1.1, colors="red", linestyle=":", label="Detected Breaths", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Envelope")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved energy envelope plot to {save_path}")


def plot_beat_alignment(beat_times: list[float] | np.ndarray, ibi: np.ndarray, tempo: float | None = None, save_path: str | Path | None = None):
    """Plot inter-beat intervals and detected beat times."""
    if not MATPLOTLIB_AVAILABLE:
        return

    save_path = Path(save_path) if save_path else RESULTS_DIR / "beat_alignment.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Beat timeline
    ax1.scatter(beat_times, [1] * len(beat_times), s=100, color="blue", label="Detected Beats")
    ax1.set_xlim(0, max(beat_times) if beat_times else 1)
    ax1.set_ylim(0.5, 1.5)
    ax1.set_ylabel("Beats")
    title = "Beat Timeline"
    if tempo:
        title += f" (Tempo: {tempo:.1f} BPM)"
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="x")

    # Inter-beat intervals
    ibi_times = np.cumsum(ibi) if len(ibi) else np.array([])
    ax2.plot(ibi_times, ibi, marker="o", linewidth=1.5, color="orange", label="Inter-Beat Intervals")
    ax2.set_xlabel("Cumulative Time (s)")
    ax2.set_ylabel("Interval (s)")
    ax2.set_title("Inter-Beat Interval Stability")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beat alignment plot to {save_path}")


def plot_note_errors(note_results: list[dict], save_path: str | Path | None = None):
    """Plot per-note estimation errors (Hz and cents)."""
    if not MATPLOTLIB_AVAILABLE:
        return

    save_path = Path(save_path) if save_path else RESULTS_DIR / "note_errors.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    targets = [n["target_hz"] for n in note_results]
    errors_hz = [n.get("error_hz") or 0.0 for n in note_results]
    errors_cents = [n.get("error_cents") or 0.0 for n in note_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(targets)), errors_hz, color="skyblue")
    ax1.set_xlabel("Note Index")
    ax1.set_ylabel("Error (Hz)")
    ax1.set_title("Per-Note Frequency Error")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(targets)), errors_cents, color="salmon")
    ax2.set_xlabel("Note Index")
    ax2.set_ylabel("Error (cents)")
    ax2.set_title("Per-Note Cents Error")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved note errors plot to {save_path}")
