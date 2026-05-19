"""Audio I/O and preprocessing utilities: loading, resampling, and frame segmentation."""

import numpy as np
import torchaudio
import torchaudio.transforms as T


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load an audio file, resample to target sample rate, and convert to mono float32.

    Args:
        path: Path to the audio file (any format supported by torchaudio).
        sr: Target sample rate in Hz. Defaults to 16000.

    Returns:
        1-D float32 numpy array of audio samples at the target sample rate.
    """
    waveform, orig_sr = torchaudio.load(path)

    if orig_sr != sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0).numpy().astype(np.float32)


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
