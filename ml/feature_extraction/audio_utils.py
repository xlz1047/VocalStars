"""Audio I/O and preprocessing utilities: loading, resampling, and frame segmentation."""

import librosa
import numpy as np

_DEFAULT_FMIN: float = 50.0
_DEFAULT_FMAX: float = 2000.0
_BREATH_THRESHOLD: float = 0.02
_BREATH_HEAD_SECS: float = 0.1


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load an audio file, resample to target sample rate, and convert to mono float32.

    Args:
        path: Path to the audio file.
        sr: Target sample rate in Hz. Defaults to 16000.

    Returns:
        1-D float32 numpy array of audio samples at the target sample rate.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def frame_audio(
    audio: np.ndarray,
    frame_len: int = 4096,
    hop_len: int = 2048,
) -> list[np.ndarray]:
    """Split audio into overlapping fixed-length frames.

    The last frame is zero-padded to ``frame_len`` if the remaining samples are
    shorter than a full frame.

    Args:
        audio: 1-D float32 numpy array of audio samples.
        frame_len: Number of samples per frame. Defaults to 4096 (256ms @ 16kHz).
        hop_len: Number of samples between successive frame starts. Defaults to 2048
            (50% overlap).

    Returns:
        List of 1-D float32 arrays each with shape ``(frame_len,)``.
    """
    frames: list[np.ndarray] = []
    n_samples = len(audio)
    start = 0

    while start < n_samples:
        end = start + frame_len
        chunk = audio[start:end]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)))
        frames.append(chunk.astype(np.float32))
        start += hop_len

    return frames


def extract_median_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = _DEFAULT_FMIN,
    fmax: float = _DEFAULT_FMAX,
) -> float:
    """Return median voiced fundamental frequency via librosa YIN.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        fmin: Minimum frequency bound in Hz.
        fmax: Maximum frequency bound in Hz.

    Returns:
        Median voiced F0 in Hz, or 0.0 if no voiced frames are detected.
    """
    f0: np.ndarray = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr)
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if len(voiced) > 0 else 0.0


def is_breath_onset(
    audio: np.ndarray,
    sr: int,
    threshold: float = _BREATH_THRESHOLD,
    head_secs: float = _BREATH_HEAD_SECS,
) -> bool:
    """Return True when the clip begins with a breath (low RMS head segment).

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate in Hz.
        threshold: RMS amplitude below which the head is considered a breath.
        head_secs: Duration of the head segment to measure in seconds.

    Returns:
        True if the head RMS is below *threshold*.
    """
    n_head = int(head_secs * sr)
    head = audio[:n_head] if len(audio) >= n_head else audio
    return float(np.sqrt(np.mean(head ** 2))) < threshold


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to the range [-1, 1].

    Silent signals (all zeros) are returned unchanged to avoid division by zero.

    Args:
        audio: 1-D float32 numpy array of audio samples.

    Returns:
        Peak-normalized float32 numpy array with values in [-1, 1].
    """
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio.copy().astype(np.float32)
    return (audio / peak).astype(np.float32)
