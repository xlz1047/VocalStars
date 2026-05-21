import numpy as np
import librosa


class RhythmExtractor:
    """Pure DSP rhythm feature extractor using librosa."""

    def onset_strength(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Compute onset strength envelope.

        Args:
            audio: Mono audio signal.
            sr: Sample rate in Hz.

        Returns:
            Onset strength envelope array.
        """
        return librosa.onset.onset_strength(y=audio, sr=sr)

    def detect_onsets(self, audio: np.ndarray, sr: int = 16000) -> list[float]:
        """Detect onset times in milliseconds.

        Args:
            audio: Mono audio signal.
            sr: Sample rate in Hz.

        Returns:
            List of onset times in milliseconds.
        """
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units="time")
        return [float(t * 1000) for t in onset_frames]

    def estimate_tempo(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Estimate tempo in BPM.

        Args:
            audio: Mono audio signal.
            sr: Sample rate in Hz.

        Returns:
            Estimated tempo in beats per minute.
        """
        tempo = librosa.beat.tempo(y=audio, sr=sr)
        return float(tempo[0])

    def compute_rhythm_regularity(self, onset_times_ms: list[float]) -> float:
        """Compute rhythm regularity score from onset times.

        Args:
            onset_times_ms: List of onset times in milliseconds.

        Returns:
            Regularity score in [0.0, 1.0]. 1.0 = perfectly regular, 0.0 = completely irregular.
        """
        if len(onset_times_ms) < 2:
            return 0.0

        ioi = np.diff(onset_times_ms)
        mean_ioi = np.mean(ioi)

        if mean_ioi == 0.0:
            return 0.0

        regularity = 1.0 - (np.std(ioi) / mean_ioi)
        return float(np.clip(regularity, 0.0, 1.0))
