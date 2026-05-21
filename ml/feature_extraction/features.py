from typing import Any
import numpy as np
import librosa


def extract_spectral_features(audio_path: str) -> dict[str, Any]:
    """Compute basic spectral features: MFCC, spectral centroid, and energy contour."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    try:
        S = np.abs(librosa.stft(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        energy = librosa.feature.rms(y=y)[0]
        return {
            "spectrogram": S.tolist(),
            "mfcc": mfcc.tolist(),
            "spectral_centroid": centroid.tolist(),
            "energy_contour": energy.tolist(),
        }
    except Exception:
        return {"spectrogram": None, "mfcc": None, "spectral_centroid": None, "energy_contour": None}
