"""MIR-1K dataset loader: 1000 Chinese karaoke clips with pitch and vocal annotations.

Expected directory structure::

    <root>/
        Wavfile/
            abjones_1_01.wav
            ...
        PitchLabel/
            abjones_1_01.pv
            ...

Each ``.wav`` file in ``Wavfile/`` is paired with a ``.pv`` file of the same
stem in ``PitchLabel/``.  The ``.pv`` files contain one pitch value per line in
Hz; ``0`` denotes an unvoiced frame.  Singer ID is derived from the filename
prefix before the first underscore (e.g. ``abjones``).
"""

from pathlib import Path

import librosa
import numpy as np

from ml.data.base_dataset import SingingDataset


class MIR1KDataset(SingingDataset):
    """Dataset loader for MIR-1K.

    Pairs each ``.wav`` with its ``.pv`` annotation file for pitch supervision.
    Falls back to librosa YIN pitch estimation when the ``.pv`` file is absent.

    Args:
        root_dir: Path to the MIR-1K root directory (containing ``Wavfile/``
            and ``PitchLabel/`` subdirectories).
        split: Split identifier (unused — kept for API consistency).
        sr: Target sample rate in Hz.
    """

    def _get_filepaths(self) -> list[dict]:
        """Pair each .wav with its matching .pv annotation file."""
        files: list[dict] = []
        wav_dir = Path(self.root_dir) / "Wavfile"
        pv_dir = Path(self.root_dir) / "PitchLabel"

        for wav_path in sorted(wav_dir.glob("*.wav")):
            singer_id = wav_path.stem.split("_")[0]
            entry: dict = {"audio_path": str(wav_path), "singer_id": singer_id}
            pv_path = pv_dir / (wav_path.stem + ".pv")
            if pv_path.exists():
                entry["pitch_path"] = str(pv_path)
            files.append(entry)

        return files

    def _extract_labels(self, audio_path: str, meta: dict) -> dict:
        """Extract labels from the .pv annotation and librosa DSP.

        Args:
            audio_path: Path to the wav file.
            meta: Metadata dict; may contain ``"pitch_path"`` and ``"singer_id"``.

        Returns:
            Dict with keys: pitch_hz, onset_frames, breath_bool, singer_id.
        """
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        if "pitch_path" in meta:
            pitches: np.ndarray = np.loadtxt(meta["pitch_path"])
            voiced = pitches[pitches > 0]
            pitch_hz = float(np.median(voiced)) if len(voiced) > 0 else 0.0
        else:
            f0: np.ndarray = librosa.yin(audio, fmin=50, fmax=2000, sr=sr)
            voiced_f0 = f0[f0 > 0]
            pitch_hz = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        onset_frames: np.ndarray = librosa.onset.onset_detect(y=audio, sr=sr)

        n_head = int(0.1 * sr)
        head = audio[:n_head] if len(audio) >= n_head else audio
        rms = float(np.sqrt(np.mean(head ** 2)))
        breath_bool = rms < 0.02

        return {
            "pitch_hz": pitch_hz,
            "onset_frames": onset_frames,
            "breath_bool": breath_bool,
            "singer_id": meta.get("singer_id", Path(audio_path).stem.split("_")[0]),
        }
