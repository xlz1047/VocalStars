"""Per-frame pitch and VAD head mapping NanoPitchEncoder embeddings to 360-bin posteriorgrams."""

import torch
import torch.nn as nn

_N_BINS: int = 360
_FMIN: float = 31.7          # Hz — B0, matches NanoPitch exactly
_CENTS_PER_BIN: float = 20.0 # 20-cent resolution over 6 octaves


class PitchHead(nn.Module):
    """Per-frame pitch classification and voice-activity head.

    Accepts backbone embeddings of shape ``(B, T, 384)`` from ``NanoPitchEncoder``
    and produces frame-level pitch logits ``(B, T, 360)`` and VAD logits
    ``(B, T, 1)``.  Bins are spaced 20 cents apart starting at 31.7 Hz (B0),
    matching NanoPitch exactly so pre-trained weights transfer directly.

    Attributes:
        FMIN: Lowest pitch bin centre in Hz.
        N_BINS: Total number of pitch bins (360).
        CENTS_PER_BIN: Width of each bin in cents (20).
        BINS_HZ: Registered buffer of shape ``(360,)`` with bin centre frequencies.
    """

    FMIN: float = _FMIN
    N_BINS: int = _N_BINS
    CENTS_PER_BIN: float = _CENTS_PER_BIN

    def __init__(self) -> None:
        super().__init__()
        self.dense_pitch = nn.Linear(384, self.N_BINS)
        self.dense_vad = nn.Linear(384, 1)

        cents = torch.arange(self.N_BINS, dtype=torch.float32) * self.CENTS_PER_BIN
        bins_hz = self.FMIN * (2.0 ** (cents / 1200.0))
        self.register_buffer("BINS_HZ", bins_hz)  # (360,)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map backbone embeddings to per-frame pitch logits and VAD logits.

        Args:
            x: Backbone embedding tensor of shape ``(B, T, 384)``.

        Returns:
            Tuple of:
                pitch_logits: ``(B, T, 360)`` raw logits (apply sigmoid for probs).
                vad_logits:   ``(B, T, 1)``   raw logits (apply sigmoid for prob).
        """
        pitch_logits = self.dense_pitch(x)  # (B, T, 360)
        vad_logits = self.dense_vad(x)      # (B, T, 1)
        return pitch_logits, vad_logits

    def logits_to_hz(self, pitch_logits: torch.Tensor) -> torch.Tensor:
        """Convert pitch logits to Hz via sigmoid-weighted mean over bin frequencies.

        Args:
            pitch_logits: ``(B, T, 360)`` or ``(B, 360)`` raw logits.

        Returns:
            Hz tensor of shape ``(B, T)`` or ``(B,)``.
        """
        probs = torch.sigmoid(pitch_logits)                       # (..., 360)
        bins = self.BINS_HZ.to(probs.device)                      # (360,)
        hz = (probs * bins).sum(-1) / (probs.sum(-1) + 1e-8)     # (...,)
        return hz

    def load_nanopitch_weights(self, checkpoint_path: str) -> None:
        """Transfer ``dense_pitch`` and ``dense_vad`` weights from a NanoPitch checkpoint.

        Args:
            checkpoint_path: Path to a NanoPitch ``.pth`` checkpoint file.
        """
        import warnings
        warnings.warn(
            "Loading checkpoint via torch.load — only use files from trusted sources.",
            RuntimeWarning,
            stacklevel=2,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        self.dense_pitch.weight.data.copy_(sd["dense_pitch.weight"])
        self.dense_pitch.bias.data.copy_(sd["dense_pitch.bias"])
        self.dense_vad.weight.data.copy_(sd["dense_vad.weight"])
        self.dense_vad.bias.data.copy_(sd["dense_vad.bias"])
        print("PitchHead: loaded dense_pitch and dense_vad weights from NanoPitch checkpoint.")
