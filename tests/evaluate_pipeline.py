"""Full-pipeline synthetic evaluator for rhythm, breath, and spectral features.

Creates synthetic audio signals with known properties and runs the
ML pipeline analyzers to report simple accuracy metrics.

Run with: `python -m tests.evaluate_pipeline`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import librosa

from ml.rhythm_analysis.rhythm_detector import analyze_rhythm
from ml.breath_analysis.breath_detector import analyze_breath
from ml.feature_extraction.features import extract_spectral_features


def synth_click_train(tempo_bpm: float, bars: int = 4, sr: int = 22050):
    sec_per_beat = 60.0 / tempo_bpm
    total_beats = int(bars * 4)
    duration = total_beats * sec_per_beat
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)
    for b in range(total_beats):
        idx = int((b * sec_per_beat) * sr)
        # short click (sine burst)
        click_len = int(0.01 * sr)
        click_t = np.linspace(0, 0.01, click_len, endpoint=False)
        y[idx:idx+click_len] += 0.9 * np.sin(2 * np.pi * 2000 * click_t) * np.hanning(click_len)
    return y, sr, duration


def synth_breathy_phrase(note_freq: float = 440.0, phrase_length: float = 6.0, breath_interval: float = 1.5, sr: int = 22050):
    # create a continuous tone with periodic low-energy breath gaps
    t = np.linspace(0, phrase_length, int(sr * phrase_length), endpoint=False)
    y = 0.6 * np.sin(2 * np.pi * note_freq * t)
    # insert breaths (short low-energy windows)
    for start in np.arange(breath_interval, phrase_length, breath_interval):
        s = int(start * sr)
        breath_len = int(0.18 * sr)
        # insert near-silence to simulate a breath gap
        y[s:s+breath_len] = 0.0
    return y, sr


def synth_tone(freq: float = 440.0, duration: float = 2.0, sr: int = 22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.7 * np.sin(2 * np.pi * freq * t), sr


def evaluate_rhythm():
    tempo_target = 100.0
    y, sr, _ = synth_click_train(tempo_target, bars=4)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    out = analyze_rhythm(tmp.name)
    tempo = out.get("tempo")
    ok = tempo is not None and abs(tempo - tempo_target) < 4.0
    return {"expected_tempo": tempo_target, "detected_tempo": tempo, "within_tolerance": ok, "raw": out}


def evaluate_breath():
    y, sr = synth_breathy_phrase()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    out = analyze_breath(tmp.name)
    # expected ~ floor(phrase_length / breath_interval) breaths
    expected = int(6.0 // 1.5)
    detected = len(out.get("breath_cycles", []))
    ok = abs(detected - expected) <= 1
    return {"expected_breaths": expected, "detected_breaths": detected, "within_tolerance": ok, "raw": out}


def evaluate_spectral():
    freq = 440.0
    y, sr = synth_tone(freq)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    out = extract_spectral_features(tmp.name)
    centroid = None
    if out.get("spectral_centroid"):
        centroid = float(np.mean(out.get("spectral_centroid")))
    ok = centroid is not None and abs(centroid - freq) < 200.0
    return {"target_freq": freq, "centroid_mean": centroid, "within_tolerance": ok, "raw": {k: (v if k!="spectrogram" else "<stft>") for k,v in out.items()}}


def run():
    print("Running full-pipeline synthetic evaluations...")
    r = evaluate_rhythm()
    b = evaluate_breath()
    s = evaluate_spectral()
    out = {"rhythm": r, "breath": b, "spectral": s}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
