"""GTSinger dataset loader: multi-technique, multi-singer English singing recordings.

Expected directory structure::

    <root>/
        English/
            <singer_id>/
                <technique>/
                    <song_name>/
                        Paired_Speech_Group/
                            <index>.wav
                            <index>.TextGrid   # present for some singers
                            <index>.json       # always present — phoneme alignments

The JSON sidecar contains a list of word dicts with ``ph_start`` / ``ph_end``
phoneme boundary times.  The technique label is derived from the directory path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

from ml.data.base_dataset import SingingDataset, split_singers_alphabetical
from ml.feature_extraction.audio_utils import extract_median_pitch, is_breath_onset

logger = logging.getLogger(__name__)

_TECHNIQUE_MAP: dict[str, int] = {
    "mixed_voice_and_falsetto": 0,
    "mixed": 0,
    "falsetto": 1,
    "breathy": 2,
    "pharyngeal": 3,
    "vibrato": 4,
    "glissando": 5,
    "normal": 6,
}

_HOP_LENGTH = 512


class GTSingerDataset(SingingDataset):
    """Dataset loader for GTSinger (English subset).

    Globs all ``*.wav`` files under ``<root>/English/`` and extracts pitch,
    phoneme-boundary onsets, breath, and technique labels.

    Args:
        root_dir: Path to the GTSinger root directory (contains ``English/``).
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Glob English/**/*.wav and return one metadata dict per audio file."""
        root = Path(self.root_dir)
        all_files: list[dict] = []
        for wav_path in sorted(root.glob("English/**/*.wav")):
            parts = wav_path.parts
            # …/English/<singer_id>/<technique>/<song_name>/Paired_Speech_Group/<n>.wav
            if len(parts) < 6:
                continue
            singer_id = parts[-5]
            technique_dir = parts[-4]
            textgrid_path = wav_path.with_suffix(".TextGrid")
            json_path = wav_path.with_suffix(".json")
            all_files.append({
                "audio_path": str(wav_path),
                "singer_id": singer_id,
                "technique_dir": technique_dir,
                "textgrid_path": str(textgrid_path) if textgrid_path.exists() else None,
                "json_path": str(json_path) if json_path.exists() else None,
            })

        all_singers = sorted({f["singer_id"] for f in all_files})
        split_singers = split_singers_alphabetical(all_singers, self.split)
        return [f for f in all_files if f["singer_id"] in split_singers]

    def _extract_labels(self, audio: np.ndarray, sr: int, meta: dict) -> dict:
        """Derive pitch, onset, breath, and technique labels for one clip.

        Args:
            audio: Already-loaded 1-D float32 audio array.
            sr: Sample rate in Hz.
            meta: Metadata dict produced by ``_get_filepaths``.

        Returns:
            Dict with keys: pitch_hz, onset_frames, breath_bool, singer_id, technique.
        """
        pitch_hz = extract_median_pitch(audio, sr)
        onset_frames = _extract_onset_frames(audio, sr, meta)
        breath_bool = is_breath_onset(audio, sr)

        technique_key = meta.get("technique_dir", "normal").lower().replace(" ", "_")
        technique_int = _TECHNIQUE_MAP.get(technique_key, _TECHNIQUE_MAP["normal"])

        return {
            "pitch_hz": pitch_hz,
            "onset_frames": onset_frames,
            "breath_bool": breath_bool,
            "singer_id": meta.get("singer_id", "unknown"),
            "technique": technique_int,
        }


def _extract_onset_frames(audio: np.ndarray, sr: int, meta: dict) -> np.ndarray:
    """Return onset frame indices from phoneme alignment or librosa fallback.

    Priority:
    1. ``ph_start`` times from the sidecar ``.TextGrid`` via the ``textgrid`` library.
    2. ``ph_start`` fields from the sidecar ``.json`` alignment file.
    3. ``librosa.onset.onset_detect`` as a last resort.

    Args:
        audio: Audio signal array.
        sr: Sample rate in Hz.
        meta: Sample metadata dict (keys: ``textgrid_path``, ``json_path``).

    Returns:
        1-D integer numpy array of onset frame indices.
    """
    tg_path = meta.get("textgrid_path")
    if tg_path:
        try:
            import textgrid  # type: ignore[import]
            tg = textgrid.TextGrid.fromFile(tg_path)
            times: list[float] = []
            for tier in tg.tiers:
                for interval in tier.intervals:
                    if interval.minTime > 0:
                        times.append(interval.minTime)
            if times:
                return np.array(
                    sorted({int(t * sr / _HOP_LENGTH) for t in times}), dtype=np.int64
                )
        except Exception as exc:
            logger.debug("TextGrid parse failed (%s), using librosa fallback", exc)

    json_path = meta.get("json_path")
    if json_path:
        try:
            with open(json_path) as fh:
                entries = json.load(fh)
            times = [
                t
                for entry in entries
                for t in entry.get("ph_start", [])
                if t > 0
            ]
            if times:
                return np.array(
                    sorted({int(t * sr / _HOP_LENGTH) for t in times}), dtype=np.int64
                )
        except Exception as exc:
            logger.debug("JSON parse failed (%s), using librosa fallback", exc)

    return librosa.onset.onset_detect(y=audio, sr=sr)


if __name__ == "__main__":
    ds = GTSingerDataset(root_dir="ml/data/raw/gtsinger", split="train")
    print(f"Train samples: {len(ds)}")
    if len(ds) == 0:
        print("No audio files found — place the dataset under ml/data/raw/gtsinger/English/")
    else:
        mel, labels = ds[0]
        print(f"Mel shape: {mel.shape}")
        print(f"Labels: {labels}")
