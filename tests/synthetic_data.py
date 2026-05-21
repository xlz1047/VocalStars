"""Expanded synthetic dataset generators for robust ML evaluation.

Includes:
- Pure sine notes (baseline)
- Microtonal offsets (±20 cents)
- Vibrato (modulated pitch)
- Noisy mixes (white + noise)
- Variable duration and amplitude envelopes
"""
from __future__ import annotations

import numpy as np
from typing import List


def synth_note(
    freq: float,
    duration: float = 1.0,
    sr: int = 22050,
    amplitude: float = 0.6,
    vibrato_freq: float = 0.0,
    vibrato_cents: float = 0.0,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Generate a single note with optional vibrato and noise.
    
    Args:
        freq: Base frequency (Hz).
        duration: Duration (seconds).
        sr: Sample rate.
        amplitude: Base amplitude.
        vibrato_freq: Vibrato LFO frequency (Hz); 0 = no vibrato.
        vibrato_cents: Vibrato depth (cents).
        noise_level: White noise mix level (0.0 = no noise).
    
    Returns:
        Audio waveform.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Apply vibrato (frequency modulation)
    f = freq
    if vibrato_freq > 0:
        # Convert cents to Hz ratio: freq * 2^(cents/1200)
        vib_ratio = 2 ** (vibrato_cents / 1200.0)
        vib_mod = 1.0 + 0.5 * (vib_ratio - 1.0) * np.sin(2 * np.pi * vibrato_freq * t)
        f = freq * vib_mod
    
    # Compute phase and waveform
    phase = 2 * np.pi * f * t
    y = amplitude * np.sin(phase)
    
    # Add white noise if requested
    if noise_level > 0:
        y += noise_level * np.random.randn(len(y))
    
    return y


def synth_melody(
    frequencies: List[float],
    durations: List[float],
    sr: int = 22050,
    amplitude: float = 0.6,
    vibrato_freq: float = 0.0,
    vibrato_cents: float = 0.0,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Generate a melody from a list of (freq, duration) pairs.
    
    All notes share the same vibrato and noise parameters.
    """
    assert len(frequencies) == len(durations)
    parts = []
    for f, d in zip(frequencies, durations):
        part = synth_note(f, d, sr, amplitude, vibrato_freq, vibrato_cents, noise_level)
        parts.append(part)
    return np.concatenate(parts)


def synth_microtonal_notes(
    base_freqs: List[float],
    cent_offsets: List[float],
    duration: float = 1.0,
    sr: int = 22050,
    amplitude: float = 0.6,
) -> np.ndarray:
    """Generate notes at microtonal offsets (for testing pitch precision).
    
    Args:
        base_freqs: Base frequencies.
        cent_offsets: Per-note cent offsets.
        duration: Duration per note.
        sr: Sample rate.
        amplitude: Amplitude.
    
    Returns:
        Concatenated audio.
    """
    assert len(base_freqs) == len(cent_offsets)
    parts = []
    for f_base, offset_cents in zip(base_freqs, cent_offsets):
        # Apply cent offset: f_actual = f_base * 2^(offset/1200)
        f_actual = f_base * (2 ** (offset_cents / 1200.0))
        part = synth_note(f_actual, duration, sr, amplitude)
        parts.append(part)
    return np.concatenate(parts)


def synth_vibrato_sweep(
    freq: float,
    duration: float = 3.0,
    sr: int = 22050,
    amplitude: float = 0.6,
    vibrato_freq_start: float = 3.0,
    vibrato_freq_end: float = 8.0,
    vibrato_cents: float = 50.0,
) -> np.ndarray:
    """Generate a single note with sweeping vibrato rate."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    vibrato_freq = np.linspace(vibrato_freq_start, vibrato_freq_end, len(t))
    vib_ratio = 2 ** (vibrato_cents / 1200.0)
    
    # Per-sample vibrato frequency
    phase_vib = 2 * np.pi * np.cumsum(vibrato_freq) / sr
    vib_mod = 1.0 + 0.5 * (vib_ratio - 1.0) * np.sin(phase_vib)
    
    phase = 2 * np.pi * freq * t
    return amplitude * np.sin(phase) * vib_mod


def synth_noisy_melody(
    frequencies: List[float],
    durations: List[float],
    sr: int = 22050,
    snr_db: float = 10.0,
) -> np.ndarray:
    """Generate a noisy melody at specified SNR."""
    signal = synth_melody(frequencies, durations, sr)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def synth_amplitude_envelope(
    freq: float,
    duration: float = 2.0,
    sr: int = 22050,
    envelope: str = "linear",
) -> np.ndarray:
    """Generate a note with an amplitude envelope (ADSR-like).
    
    Args:
        freq: Frequency.
        duration: Duration.
        sr: Sample rate.
        envelope: "linear", "exponential", or "adsr".
    
    Returns:
        Audio with envelope applied.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.6 * np.sin(2 * np.pi * freq * t)
    
    if envelope == "linear":
        env = np.linspace(0, 1, len(t) // 2)
        env = np.concatenate([env, env[::-1]])
    elif envelope == "exponential":
        env = np.exp(-2 * t / duration)
    elif envelope == "adsr":
        # Simple ADSR: attack=10%, decay=20%, sustain=50%, release=20%
        attack_len = int(0.1 * len(t))
        decay_len = int(0.2 * len(t))
        sustain_len = int(0.5 * len(t))
        release_len = len(t) - attack_len - decay_len - sustain_len
        env = np.concatenate([
            np.linspace(0, 1, attack_len),
            np.linspace(1, 0.7, decay_len),
            np.ones(sustain_len) * 0.7,
            np.linspace(0.7, 0, release_len)
        ])
    else:
        env = np.ones_like(t)
    
    return y * env
