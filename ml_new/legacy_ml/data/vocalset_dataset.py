"""VocalSet dataset loader: 20-singer professional vocal technique recordings.

Expected directory structure::

    <root>/
        FULL/
            female1/
                <technique>/
                    *.wav
            ...
            male11/
                <technique>/
                    *.wav
        train_singers.rtfd/
            TXT.rtf
        test_singers.rtfd/
            TXT.rtf

Singer/split assignment is read from the .rtfd split files; if those are
absent or unreadable a deterministic 80/10/10 alphabetical fallback is used.
"""

from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np

from ml_new.legacy_ml.data.base_dataset import SingingDataset, split_singers_alphabetical
from ml_new.legacy_ml.feature_extraction.audio_utils import extract_median_pitch, is_breath_onset


class VocalSetDataset(SingingDataset):
    """Dataset loader for VocalSet.

    Globs all ``*.wav`` files under ``<root>/FULL/<singer>/<technique>/`` and
    extracts pitch, onsets, and breath labels entirely via DSP.

    Args:
        root_dir: Path to the VocalSet root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Glob FULL/<singer>/<category>/<technique>/*.wav and filter by split."""
        root = Path(self.root_dir)
        all_files: list[dict] = []
        for wav_path in sorted(root.glob("FULL/*/*/*/*.wav")):
            singer_id = wav_path.parts[-4]   # e.g. "female1"
            technique = wav_path.parts[-2]   # e.g. "belt"
            all_files.append({
                "audio_path": str(wav_path),
                "singer_id": singer_id,
                "technique": technique,
            })

        split_singers = self._resolve_split_singers(
            root, sorted({f["singer_id"] for f in all_files})
        )
        return [f for f in all_files if f["singer_id"] in split_singers]

    def _resolve_split_singers(self, root: Path, all_singers: list[str]) -> set[str]:
        """Return the singer IDs that belong to this dataset split.

        Tries to parse the .rtfd split files that ship with VocalSet.  Falls
        back to an 80/10/10 alphabetical partition when those files are missing
        or cannot be decoded.

        Args:
            root: VocalSet root directory.
            all_singers: Sorted list of all singer IDs found on disk.

        Returns:
            Set of singer ID strings assigned to ``self.split``.
        """
        train_rtf = root / "train_singers.rtfd" / "TXT.rtf"
        test_rtf = root / "test_singers.rtfd" / "TXT.rtf"

        if train_rtf.exists() and test_rtf.exists():
            try:
                train_singers = set(_parse_rtf_singers(train_rtf))
                test_singers = set(_parse_rtf_singers(test_rtf))
                val_singers = set(all_singers) - train_singers - test_singers
                mapping = {"train": train_singers, "val": val_singers, "test": test_singers}
                return mapping[self.split]
            except Exception:
                pass  # fall through to deterministic split

        return split_singers_alphabetical(all_singers, self.split)

    def _extract_labels(self, audio: np.ndarray, sr: int, meta: dict) -> dict:
        """Extract labels using librosa DSP heuristics.

        Args:
            audio: Already-loaded 1-D float32 audio array.
            sr: Sample rate in Hz.
            meta: Metadata dict with keys ``"singer_id"`` and ``"technique"``.

        Returns:
            Dict with keys: pitch_hz, onset_frames, breath_bool, singer_id, technique.
        """
        pitch_hz = extract_median_pitch(audio, sr)
        onset_frames: np.ndarray = librosa.onset.onset_detect(y=audio, sr=sr)
        breath_bool = is_breath_onset(audio, sr)

        return {
            "pitch_hz": pitch_hz,
            "onset_frames": onset_frames,
            "breath_bool": breath_bool,
            "singer_id": meta.get("singer_id", "unknown"),
            "technique": meta.get("technique", "unknown"),
        }


def _parse_rtf_singers(path: Path) -> list[str]:
    """Extract singer-name tokens from a minimal RTF file.

    Strips RTF control words and returns only tokens that look like VocalSet
    singer IDs (``female<N>`` or ``male<N>``).

    Args:
        path: Path to a ``TXT.rtf`` file.

    Returns:
        List of singer ID strings found in the file.
    """
    raw = path.read_bytes().decode("utf-8", errors="ignore")
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", raw)        # hex escape sequences
    text = re.sub(r"\\[a-zA-Z]+-?\d*[ ]?", " ", text)   # control words
    text = re.sub(r"[{}\\]", " ", text)                  # braces and backslashes
    return [tok for tok in text.split() if re.match(r"^(female|male)\d+$", tok)]


if __name__ == "__main__":
    ds = VocalSetDataset(root_dir="ml/data/raw/vocalset", split="train")
    print(f"Train samples: {len(ds)}")
    if len(ds) == 0:
        print("No audio files found — place the dataset under ml/data/raw/vocalset/FULL/")
    else:
        mel, labels = ds[0]
        print(f"Mel shape: {mel.shape}")
        print(f"Labels: {labels}")
