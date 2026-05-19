"""Shared convolutional backbone that produces frame-level embeddings from mel input."""

import torch
import torch.nn as nn


def _dw_sep_block(in_ch: int, out_ch: int, stride: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AudioEncoder(nn.Module):
    """Depthwise-separable convolutional backbone for log-mel spectrogram embeddings.

    Maps (batch, 1, 128, T) log-mel tensors to frame-level embeddings of shape
    (batch, 256, T//4) via four depthwise-separable conv blocks followed by
    frequency-axis average pooling.

    Blocks 1–2 apply temporal stride 2, reducing T to T//4 across two steps.
    Blocks 3–4 refine channel depth without further downsampling.
    The frequency dimension (128) is collapsed to 1 by adaptive average pooling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _dw_sep_block(1,   32,  stride=(1, 2)),
            _dw_sep_block(32,  64,  stride=(1, 2)),
            _dw_sep_block(64,  128, stride=(1, 1)),
            _dw_sep_block(128, 256, stride=(1, 1)),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        total = sum(p.numel() for p in self.parameters())
        assert total < 800_000, f"Backbone too large: {total} params"
        self.param_count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of log-mel spectrograms into frame-level embeddings.

        Args:
            x: Log-mel spectrogram tensor of shape (batch, 1, 128, T).

        Returns:
            Frame-level embeddings of shape (batch, 256, T//4).
        """
        x = self.blocks(x)      # (batch, 256, 128, T//4)
        x = self.freq_pool(x)   # (batch, 256, 1, T//4)
        return x.squeeze(2)     # (batch, 256, T//4)
