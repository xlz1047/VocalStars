"""Evaluation script: per-task metrics (RPA, OA, F1) computed on held-out validation sets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ml._model.voice_coach import VoiceCoachModel
from ml.pitch_detection.model import PitchHead
from ml.training.losses import MultiTaskLoss

_BINS_HZ: torch.Tensor = PitchHead.FMIN * (
    2 ** (torch.arange(PitchHead.N_BINS, dtype=torch.float32) * PitchHead.CENTS_PER_BIN / 1200.0)
)

_CENTS_TOLERANCE = 50.0

_LOSS_FN = MultiTaskLoss()


def _pitch_hz_from_logits(logits: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    """Weighted-mean frequency estimate from pitch bin logits.

    Args:
        logits: Tensor of shape (batch, 360).
        top_k: Number of highest-probability bins used for the weighted mean.

    Returns:
        Float tensor of shape (batch,) with estimated Hz values.
    """
    probs = torch.sigmoid(logits)
    bins = _BINS_HZ.to(logits.device)
    top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
    pitch_hz = (bins[top_idx] * top_vals).sum(dim=-1) / top_vals.sum(dim=-1).clamp(min=1e-8)
    return pitch_hz


def _hz_to_pitch_bins(pitch_hz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Soft Gaussian pitch-bin targets identical to those used in training."""
    bins_hz = _BINS_HZ.to(device)
    hz_safe = pitch_hz.clamp(min=1e-6).unsqueeze(1)
    cents = 1200.0 * torch.log2(bins_hz.unsqueeze(0) / hz_safe)
    targets = torch.exp(-0.5 * (cents / 20.0) ** 2)
    voiced = (pitch_hz > 0).float().unsqueeze(1)
    return targets * voiced


def _onset_targets_for_preds(onset_raw: torch.Tensor, T_out: int) -> torch.Tensor:
    """Downsample dense onset targets to match backbone output length T_out."""
    T_in = onset_raw.shape[-1]
    stride = max(1, T_in // T_out)
    return onset_raw[:, ::stride][:, :T_out]


@torch.no_grad()
def evaluate_model(
    model: VoiceCoachModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute pitch, onset, and breath metrics on a held-out dataset.

    Metrics:
        pitch_rpa:  Raw pitch accuracy — fraction of voiced frames where the
                    predicted pitch is within 50 cents of the target.
        pitch_rca:  Raw chroma accuracy — same as RPA but octave errors are
                    forgiven (pitch correct modulo octave).
        onset_f1:   F1 score on per-frame onset detection (threshold 0.5).
        breath_acc: Binary accuracy on the breath event flag.
        overall:    Weighted mean: ``pitch_rpa × 0.4 + onset_f1 × 0.3 + breath_acc × 0.3``.
        pitch_loss: Mean pitch BCE-with-logits loss (used by train loop to track best model).

    Args:
        model:      Trained ``VoiceCoachModel``.
        dataloader: DataLoader yielding (mel, labels) batches from ``_collate_fn``.
        device:     Torch device.

    Returns:
        Dict mapping metric name to scalar float.
    """
    model.eval()

    total_pitch_correct = 0.0
    total_chroma_correct = 0.0
    total_voiced = 0.0

    tp_onset = 0.0
    fp_onset = 0.0
    fn_onset = 0.0

    breath_correct = 0.0
    breath_total = 0.0

    vad_correct = 0.0
    vad_total = 0.0

    pitch_loss_sum = 0.0
    n_batches = 0

    for mel, targets_raw in dataloader:
        mel = mel.to(device, non_blocking=True)
        pitch_hz_gt = targets_raw["pitch_hz"].to(device)
        onset_t = targets_raw["onset_targets"].to(device)
        breath_t = targets_raw["breath_target"].to(device)

        preds = model(mel)
        pitch_logits: torch.Tensor = preds["pitch_logits"]
        onset_probs: torch.Tensor  = preds["onset_probs"]
        breath_prob: torch.Tensor  = preds["breath_prob"]

        # ── VAD accuracy ──────────────────────────────────────────────────────
        if "vad_logits" in preds:
            vad_mean = torch.sigmoid(preds["vad_logits"]).squeeze(-1).mean(dim=1)  # (B,)
            voiced_proxy = (pitch_hz_gt > 0).float()
            vad_correct += (vad_mean > 0.5).float().eq(voiced_proxy).sum().item()
            vad_total += float(voiced_proxy.size(0))

        # ── pitch metrics ─────────────────────────────────────────────────────
        # Pick the single most-confident frame per clip to avoid dilution from
        # unvoiced frames whose logits are near zero (sigmoid ≈ 0.5 uniform).
        if pitch_logits.dim() == 3:
            B_eval = pitch_logits.size(0)
            probs_all = torch.sigmoid(pitch_logits)           # (B, T, 360)
            confidence = probs_all.max(dim=-1).values         # (B, T)
            best_frame = confidence.argmax(dim=1)             # (B,)
            pitch_logits_eval = pitch_logits[
                torch.arange(B_eval, device=pitch_logits.device), best_frame
            ]                                                 # (B, 360)
        else:
            pitch_logits_eval = pitch_logits
        pitch_bins_gt = _hz_to_pitch_bins(pitch_hz_gt, device)
        pitch_loss_sum += _LOSS_FN._pitch_loss_fn(pitch_logits_eval, pitch_bins_gt).item()

        pred_hz = _pitch_hz_from_logits(pitch_logits_eval)  # (batch,)
        voiced_mask = pitch_hz_gt > 0

        if voiced_mask.any():
            gt_v  = pitch_hz_gt[voiced_mask].clamp(min=1e-6)
            pr_v  = pred_hz[voiced_mask].clamp(min=1e-6)
            cents = torch.abs(1200.0 * torch.log2(pr_v / gt_v))
            total_pitch_correct += (cents < _CENTS_TOLERANCE).float().sum().item()

            # Chroma: reduce both to same octave relative to C1 (32.7 Hz)
            gt_chroma = gt_v * (2 ** (-torch.floor(torch.log2(gt_v / 32.7))))
            pr_chroma = pr_v * (2 ** (-torch.floor(torch.log2(pr_v / 32.7))))
            chroma_cents = torch.abs(1200.0 * torch.log2(pr_chroma / gt_chroma.clamp(min=1e-6)))
            total_chroma_correct += (chroma_cents < _CENTS_TOLERANCE).float().sum().item()
            total_voiced += voiced_mask.float().sum().item()

        # ── onset F1 ──────────────────────────────────────────────────────────
        T_out = onset_probs.shape[-1]
        onset_t_ds = _onset_targets_for_preds(onset_t, T_out)
        onset_pred = (onset_probs.view(mel.size(0), -1) > 0.5).float()
        onset_gt   = (onset_t_ds > 0.5).float()

        tp_onset += (onset_pred * onset_gt).sum().item()
        fp_onset += (onset_pred * (1.0 - onset_gt)).sum().item()
        fn_onset += ((1.0 - onset_pred) * onset_gt).sum().item()

        # ── breath accuracy ───────────────────────────────────────────────────
        breath_pred = (breath_prob.view(-1) > 0.5).float()
        breath_correct += (breath_pred == breath_t.float()).float().sum().item()
        breath_total += float(breath_t.size(0))

        n_batches += 1

    # ── aggregate ─────────────────────────────────────────────────────────────
    pitch_rpa = total_pitch_correct / max(total_voiced, 1.0)
    pitch_rca = total_chroma_correct / max(total_voiced, 1.0)

    precision = tp_onset / max(tp_onset + fp_onset, 1.0)
    recall    = tp_onset / max(tp_onset + fn_onset, 1.0)
    onset_f1  = (
        2.0 * precision * recall / max(precision + recall, 1e-8)
    )

    breath_acc = breath_correct / max(breath_total, 1.0)
    vad_acc    = vad_correct / max(vad_total, 1.0)
    overall    = pitch_rpa * 0.4 + onset_f1 * 0.3 + breath_acc * 0.3
    pitch_loss = pitch_loss_sum / max(n_batches, 1)

    return {
        "pitch_rpa":  pitch_rpa,
        "pitch_rca":  pitch_rca,
        "onset_f1":   onset_f1,
        "breath_acc": breath_acc,
        "vad_acc":    vad_acc,
        "overall":    overall,
        "pitch_loss": pitch_loss,
    }
