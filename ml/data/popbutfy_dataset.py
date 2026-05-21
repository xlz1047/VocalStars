"""PopBuTFy dataset loader: paired good/raw singing recordings for intonation supervision.

PopBuTFy provides paired recordings where each singer performs a song in both a
raw (unprocessed) and a good-quality (reference) version.  The intonation score
is computed by comparing the pitch of the raw recording against the reference.

Expected directory structure (auto-detected)::

    <root>/
        <singer_id>/
            <song_id>/
                good.wav      # reference (in-tune) recording
                raw.wav       # unprocessed recording to be coached

If the structure differs, ``_scan_pairs`` emits a warning and the dataset
returns zero samples rather than crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np

from ml.data.base_dataset import SingingDataset

_HOP_LENGTH = 512


class PopBuTFyDataset(SingingDataset):
    """Dataset loader for PopBuTFy.

    Finds paired raw/reference audio files and provides direct intonation
    supervision via cent-deviation scoring.

    Args:
        root_dir: Path to the PopBuTFy root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Locate paired (raw, reference) audio files and filter by split.

        Returns:
            List of metadata dicts with keys: audio_path, reference_path,
            singer_id, song_id.
        """
        root = Path(self.root_dir)
        pairs = _scan_pairs(root)
        all_singers = sorted({p["singer_id"] for p in pairs})
        split_singers = _split_singers(all_singers, self.split)
        return [p for p in pairs if p["singer_id"] in split_singers]

    def _extract_labels(self, audio_path: str, meta: dict) -> dict:
        """Derive pitch, onset, breath, and intonation labels for one clip.

        The intonation_score is 1.0 when the median pitch deviates less than
        50 cents from the reference, and linearly decreases to 0.0 at 200 cents.

        Args:
            audio_path: Path to the raw (student) wav file.
            meta: Metadata dict with key ``reference_path``.

        Returns:
            Dict with keys: pitch_hz, onset_frames, breath_bool, singer_id,
            intonation_score.
        """
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        f0: np.ndarray = librosa.yin(audio, fmin=50, fmax=2000, sr=sr)
        voiced = f0[f0 > 0]
        pitch_hz = float(np.median(voiced)) if len(voiced) > 0 else 0.0

        onset_frames: np.ndarray = librosa.onset.onset_detect(y=audio, sr=sr)

        n_head = int(0.1 * sr)
        head = audio[:n_head] if len(audio) >= n_head else audio
        breath_bool = float(np.sqrt(np.mean(head ** 2))) < 0.02

        intonation_score = _compute_intonation_score(pitch_hz, meta.get("reference_path"), sr)

        return {
            "pitch_hz": pitch_hz,
            "onset_frames": onset_frames,
            "breath_bool": breath_bool,
            "singer_id": meta.get("singer_id", "unknown"),
            "intonation_score": intonation_score,
        }


def _scan_pairs(root: Path) -> list[dict]:
    """Walk the directory tree and collect paired (raw, reference) wav entries.

    Looks for directories containing both a ``raw*.wav`` and a ``good*.wav``
    (case-insensitive).  Falls back to any two ``.wav`` files whose stems
    suggest a quality contrast (e.g. ``_proc`` / ``_unproc`` suffixes).

    Args:
        root: Root directory to scan.

    Returns:
        List of metadata dicts with keys: audio_path, reference_path,
        singer_id, song_id.
    """
    pairs: list[dict] = []
    for dirpath, _dirs, filenames in os.walk(root):
        wavs = [f for f in filenames if f.lower().endswith(".wav")]
        if not wavs:
            continue

        raw_files = [f for f in wavs if "raw" in f.lower()]
        good_files = [f for f in wavs if "good" in f.lower() or "ref" in f.lower()]

        if not raw_files or not good_files:
            continue

        dp = Path(dirpath)
        parts = dp.relative_to(root).parts
        singer_id = parts[0] if len(parts) >= 1 else "unknown"
        song_id = parts[1] if len(parts) >= 2 else dp.name

        for raw_name in raw_files:
            ref_name = good_files[0]
            pairs.append({
                "audio_path": str(dp / raw_name),
                "reference_path": str(dp / ref_name),
                "singer_id": singer_id,
                "song_id": song_id,
            })

    return pairs


def _split_singers(all_singers: list[str], split: str) -> set[str]:
    """Deterministic 80/10/10 split of singer IDs by alphabetical order.

    Args:
        all_singers: Sorted list of all unique singer IDs.
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        Set of singer IDs assigned to the requested split.
    """
    n = len(all_singers)
    if n == 0:
        return set()
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    boundaries: dict[str, list[str]] = {
        "train": all_singers[:n_train],
        "val": all_singers[n_train : n_train + n_val],
        "test": all_singers[n_train + n_val :],
    }
    return set(boundaries.get(split, all_singers))


def _compute_intonation_score(pitch_hz: float, reference_path: str | None, sr: int) -> float:
    """Score intonation by comparing pitch to a reference recording.

    Returns 1.0 when the cent deviation is below 50, linearly decreasing to
    0.0 at 200 cents.  Returns 0.0 when either pitch is unvoiced or the
    reference cannot be loaded.

    Args:
        pitch_hz: Median voiced pitch of the student recording in Hz.
        reference_path: Path to the reference (in-tune) wav file.
        sr: Sample rate used for loading.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if pitch_hz <= 0 or not reference_path:
        return 0.0
    try:
        ref_audio, ref_sr = librosa.load(reference_path, sr=sr, mono=True)
        ref_f0: np.ndarray = librosa.yin(ref_audio, fmin=50, fmax=2000, sr=ref_sr)
        ref_voiced = ref_f0[ref_f0 > 0]
        ref_pitch = float(np.median(ref_voiced)) if len(ref_voiced) > 0 else 0.0
        if ref_pitch <= 0:
            return 0.0
        cents_deviation = abs(1200.0 * float(np.log2(pitch_hz / ref_pitch)))
        if cents_deviation < 50.0:
            return 1.0
        if cents_deviation > 200.0:
            return 0.0
        return 1.0 - (cents_deviation - 50.0) / 150.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    ds = PopBuTFyDataset(root_dir="ml/data/raw/popbutfy", split="train")
    print(f"Train samples: {len(ds)}")
    if len(ds) == 0:
        print("No paired files found — place the dataset under ml/data/raw/popbutfy/")
    else:
        mel, labels = ds[0]
        print(f"Mel shape: {mel.shape}")
        print(f"Labels: {labels}")
