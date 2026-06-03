"""NanoPitchEncoder: causal Conv1d + 3-layer GRU backbone for frame-level audio embeddings.

Architecture (adapted from NanoPitch / RNNoise):
    Input: (B, T, 40) log-mel, time-major
    Conv1d(40→64, k=3, causal) + tanh   →  conv_out (B, T, 96)  [after conv2]
    GRU1(96→96), GRU2(96→96), GRU3(96→96)
    cat([conv_out, g1, g2, g3], dim=-1)  →  (B, T, 384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NanoPitchEncoder(nn.Module):
    """Causal convolutional + GRU backbone for real-time pitch-aware audio encoding.

    Processes a log-mel spectrogram of shape ``(B, T, n_mels)`` (time-major) and
    produces frame-level embeddings of shape ``(B, T, 384)`` via two stacked causal
    Conv1d layers followed by three GRU layers whose outputs are concatenated.

    The causal padding ensures ``output[t]`` depends only on ``input[t-2..t]``, making
    the model suitable for streaming inference without added latency.

    Args:
        n_mels: Number of input mel bands. Must match the MelExtractor setting (40).
        cond_size: Width of the first Conv1d layer (64).
        gru_size: Hidden size of each GRU layer (96).
    """

    def __init__(self, n_mels: int = 40, cond_size: int = 64, gru_size: int = 96) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.cond_size = cond_size
        self.gru_size = gru_size

        self.conv1 = nn.Conv1d(n_mels, cond_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=3, padding=0)

        self.gru1 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru2 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru3 = nn.GRU(gru_size, gru_size, batch_first=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for GRU recurrent weight matrices."""
        for module in self.modules():
            if isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)

    def forward(
        self, x: torch.Tensor, states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a batch of log-mel spectrograms into frame-level embeddings.

        Args:
            x: Log-mel tensor of shape ``(B, T, n_mels)``.
            states: Optional list of three GRU hidden states ``[(1,B,96), ...]``.
                    Pass ``None`` to start from zeros.

        Returns:
            Tuple of:
                features: ``(B, T, 384)`` concatenation of conv_out and all GRU outputs.
                new_states: List of three updated GRU hidden states.
        """
        B = x.size(0)
        device = x.device

        if states is None:
            h1 = torch.zeros(1, B, self.gru_size, device=device)
            h2 = torch.zeros(1, B, self.gru_size, device=device)
            h3 = torch.zeros(1, B, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        # Conv block: (B, T, C) → (B, C, T) → causal pad → conv → tanh → back
        c = x.permute(0, 2, 1)                         # (B, 40, T)
        c = F.pad(c, (2, 0))                            # (B, 40, T+2)
        c = torch.tanh(self.conv1(c))                   # (B, 64, T)
        c = F.pad(c, (2, 0))                            # (B, 64, T+2)
        c = torch.tanh(self.conv2(c))                   # (B, 96, T)
        conv_out = c.permute(0, 2, 1)                   # (B, T, 96)

        g1, h1 = self.gru1(conv_out, h1)               # (B, T, 96)
        g2, h2 = self.gru2(g1, h2)                     # (B, T, 96)
        g3, h3 = self.gru3(g2, h3)                     # (B, T, 96)

        features = torch.cat([conv_out, g1, g2, g3], dim=-1)  # (B, T, 384)
        return features, [h1, h2, h3]

    def init_streaming_state(self, batch_size: int = 1, device: str = "cpu") -> dict:
        """Create zeroed streaming buffers for frame-by-frame inference.

        Args:
            batch_size: Number of parallel streams.
            device: Torch device string.

        Returns:
            Dict with keys ``conv1_buf``, ``conv2_buf``, ``gru_states``.
        """
        return {
            "conv1_buf": torch.zeros(batch_size, 2, self.n_mels, device=device),
            "conv2_buf": torch.zeros(batch_size, 2, self.cond_size, device=device),
            "gru_states": [
                torch.zeros(1, batch_size, self.gru_size, device=device),
                torch.zeros(1, batch_size, self.gru_size, device=device),
                torch.zeros(1, batch_size, self.gru_size, device=device),
            ],
        }

    @classmethod
    def from_nanopitch_checkpoint(cls, checkpoint_path: str) -> "NanoPitchEncoder":
        """Build an encoder and initialise its weights from a NanoPitch ``.pth`` file.

        Transfers conv1, conv2, and all three GRU layers.  The output heads
        (``dense_pitch``, ``dense_vad``) in the checkpoint are intentionally
        excluded — those belong to ``PitchHead``.

        Args:
            checkpoint_path: Path to a NanoPitch checkpoint saved by ``train.py``.

        Returns:
            Initialised ``NanoPitchEncoder`` instance.
        """
        import warnings
        warnings.warn(
            "Loading checkpoint via torch.load — only use files from trusted sources.",
            RuntimeWarning,
            stacklevel=2,
        )
        encoder = cls()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        backbone_keys = [
            "conv1.weight", "conv1.bias",
            "conv2.weight", "conv2.bias",
            "gru1.weight_ih_l0", "gru1.weight_hh_l0", "gru1.bias_ih_l0", "gru1.bias_hh_l0",
            "gru2.weight_ih_l0", "gru2.weight_hh_l0", "gru2.bias_ih_l0", "gru2.bias_hh_l0",
            "gru3.weight_ih_l0", "gru3.weight_hh_l0", "gru3.bias_ih_l0", "gru3.bias_hh_l0",
        ]
        filtered = {k: sd[k] for k in backbone_keys if k in sd}
        result = encoder.load_state_dict(filtered, strict=False)
        print(f"NanoPitchEncoder: loaded {len(filtered)}/16 backbone tensors. "
              f"Missing: {result.missing_keys}")
        return encoder
