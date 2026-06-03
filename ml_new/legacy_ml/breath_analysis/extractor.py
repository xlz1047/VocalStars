"""Pure DSP breath feature extractor: energy envelope and spectral breath band analysis."""

import numpy as np
import librosa


class BreathExtractor:
    """Extracts DSP features from a single audio frame for breath classification.

    Computes energy, zero-crossing rate, spectral centroid, and MFCCs, then
    applies rule-based heuristics to classify the frame as silence, breath, or singing.
    """

    def extract_features(self, audio_frame: np.ndarray, sr: int = 16000) -> dict:
        """Compute acoustic features for one audio frame.

        Args:
            audio_frame: 1-D float32 numpy array of audio samples.
            sr: Sample rate of the audio in Hz.

        Returns:
            Dict with keys:
                rms_energy (float): Root mean square energy of the frame.
                zero_crossing_rate (float): ZCR normalized by frame length.
                spectral_centroid (float): Spectral centroid in Hz.
                mfcc_mean (np.ndarray): Mean of 13 MFCCs across time, shape (13,).
        """
        rms_energy = float(np.sqrt(np.mean(audio_frame ** 2)))

        signs = np.sign(audio_frame)
        zcr = float(np.sum(np.abs(np.diff(signs))) / 2 / len(audio_frame))

        centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=sr)
        spectral_centroid = float(np.mean(centroid))

        mfccs = librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1)

        return {
            "rms_energy": rms_energy,
            "zero_crossing_rate": zcr,
            "spectral_centroid": spectral_centroid,
            "mfcc_mean": mfcc_mean,
        }

    def classify_heuristic(self, features: dict) -> tuple[str, float]:
        """Classify a frame as silence, breath, or singing based on acoustic features.

        Args:
            features: Dict as returned by ``extract_features``.

        Returns:
            Tuple of (label, confidence) where label is one of
            ``"silence"``, ``"breath"``, or ``"singing"`` and confidence
            is a float in [0.0, 1.0] reflecting how strongly the rule fires.
        """
        rms = features["rms_energy"]
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid"]

        if rms < 0.01:
            conf = float(np.clip(1.0 - rms / 0.01, 0.0, 1.0))
            return "silence", conf

        if rms < 0.05 and zcr > 0.1 and centroid < 2000:
            rms_margin = (0.05 - rms) / 0.05
            zcr_margin = float(np.clip((zcr - 0.1) / 0.4, 0.0, 1.0))
            centroid_margin = float(np.clip((2000.0 - centroid) / 2000.0, 0.0, 1.0))
            conf = float(np.clip((rms_margin + zcr_margin + centroid_margin) / 3.0, 0.0, 1.0))
            return "breath", conf

        above_breath = float(np.clip((rms - 0.05) / 0.45, 0.0, 1.0))
        conf = float(np.clip(0.5 + above_breath * 0.5, 0.0, 1.0))
        return "singing", conf
