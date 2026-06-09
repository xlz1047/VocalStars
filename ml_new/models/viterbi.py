"""Viterbi decoder for pitch tracking post-processing.

States: 0..N_BINS-1 = voiced pitch bins; N_BINS = unvoiced.
"""

from __future__ import annotations

import numpy as np
import torch


def build_log_trans(
    n_bins: int,
    sigma_bins: float = 2.0,
    voice_change_penalty: float = 5.0,
) -> np.ndarray:
    """Build log transition matrix for (n_bins + 1) states.

    Voiced→voiced: Gaussian-shaped log probability centred on 0 pitch jump.
    Voiced↔unvoiced: fixed penalty.
    Unvoiced→unvoiced: 0 (log prob = 0, i.e. prob = 1 before normalisation).

    Args:
        n_bins: Number of pitch bins (voiced states).
        sigma_bins: Gaussian width in bins for voiced→voiced transitions.
        voice_change_penalty: Log-prob penalty for voiced↔unvoiced switches.

    Returns:
        ``(n_bins+1, n_bins+1)`` log transition matrix.
        Entry [i, j] = log P(next=j | current=i).
    """
    S = n_bins + 1  # total states; last state = unvoiced
    log_trans = np.full((S, S), -np.inf, dtype=np.float32)

    # Voiced → voiced: Gaussian over bin distance
    bins = np.arange(n_bins, dtype=np.float32)
    for i in range(n_bins):
        diff = bins - i
        log_p = -0.5 * (diff / sigma_bins) ** 2
        log_p -= np.logaddexp.reduce(log_p)  # normalise
        log_trans[i, :n_bins] = log_p

    # Voiced → unvoiced
    log_trans[:n_bins, n_bins] = -voice_change_penalty

    # Unvoiced → voiced
    log_trans[n_bins, :n_bins] = -voice_change_penalty

    # Unvoiced → unvoiced
    log_trans[n_bins, n_bins] = 0.0

    return log_trans


def viterbi_decode(log_obs: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Vectorised Viterbi algorithm.

    Args:
        log_obs: ``(T, S)`` log emission probabilities.
        log_trans: ``(S, S)`` log transition probabilities, entry [i,j] = log P(j|i).

    Returns:
        ``(T,)`` integer array of most-likely state indices.
    """
    T, S = log_obs.shape
    V = np.full((T, S), -np.inf, dtype=np.float32)
    ptr = np.zeros((T, S), dtype=np.int32)

    V[0] = log_obs[0]

    for t in range(1, T):
        # scores[i, j] = V[t-1, i] + log_trans[i, j]
        scores = V[t - 1, :, None] + log_trans  # (S, S)
        ptr[t] = scores.argmax(axis=0)           # (S,)
        V[t] = scores.max(axis=0) + log_obs[t]  # (S,)

    # Backtrack
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = V[T - 1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return path


def pitch_viterbi(
    pitch_logits: torch.Tensor,
    voiced_prob: torch.Tensor,
    sigma_bins: float = 2.0,
    voice_change_penalty: float = 5.0,
    voiced_threshold: float = 0.5,
    pitch_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Viterbi decoding to a single sequence.

    Args:
        pitch_logits: ``(T, n_bins)`` or ``(1, T, n_bins)`` pitch logits.
        voiced_prob: ``(T,)`` or ``(1, T)`` predicted voicing probability.
        sigma_bins: Gaussian transition width (bins).
        voice_change_penalty: Penalty for voiced↔unvoiced transitions (full mode only).
        voiced_threshold: Decision boundary for voiced/unvoiced.
        pitch_only: If True, keep argmax voicing and only smooth pitch within voiced
            segments via Viterbi. If False, use joint voiced+pitch Viterbi.

    Returns:
        pred_bins: ``(T,)`` predicted pitch bin indices (−1 = unvoiced).
        pred_voiced: ``(T,)`` boolean array.
    """
    if pitch_logits.dim() == 3:
        pitch_logits = pitch_logits.squeeze(0)
    if voiced_prob.dim() == 2:
        voiced_prob = voiced_prob.squeeze(0)

    T, n_bins = pitch_logits.shape
    logits_np = pitch_logits.detach().cpu().float().numpy()
    voiced_np = voiced_prob.detach().cpu().float().numpy()

    if pitch_only:
        # Keep argmax voicing; only Viterbi-smooth pitch on voiced frames.
        pred_voiced = voiced_np >= voiced_threshold
        pred_bins = np.full(T, -1, dtype=np.int32)

        # Build voiced-only transition matrix (n_bins states)
        log_trans_p = build_log_trans(n_bins, sigma_bins, voice_change_penalty)[:n_bins, :n_bins]
        # Re-normalise rows (drop unvoiced column)
        for i in range(n_bins):
            row = log_trans_p[i]
            mx = np.logaddexp.reduce(row)
            log_trans_p[i] = row - mx

        import torch.nn.functional as F
        log_pitch = F.log_softmax(torch.from_numpy(logits_np), dim=-1).numpy()  # (T, n_bins)

        # Find voiced segments and apply Viterbi within each
        voiced_idx = np.where(pred_voiced)[0]
        if len(voiced_idx) > 0:
            # Group consecutive voiced frames
            segments = []
            start = voiced_idx[0]
            prev = voiced_idx[0]
            for idx in voiced_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    segments.append((start, prev + 1))
                    start = idx
                    prev = idx
            segments.append((start, prev + 1))

            for s, e in segments:
                if e - s == 1:
                    pred_bins[s] = logits_np[s].argmax()
                else:
                    seg_path = viterbi_decode(log_pitch[s:e], log_trans_p)
                    pred_bins[s:e] = seg_path

        return pred_bins, pred_voiced

    # Joint voiced+pitch Viterbi
    S = n_bins + 1
    import torch.nn.functional as F
    log_pitch = F.log_softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    log_voiced = np.log(np.clip(voiced_np, 1e-7, 1 - 1e-7))
    log_unvoiced = np.log(np.clip(1 - voiced_np, 1e-7, 1 - 1e-7))

    log_obs = np.full((T, S), -np.inf, dtype=np.float32)
    log_obs[:, :n_bins] = log_pitch + log_voiced[:, None]
    log_obs[:, n_bins] = log_unvoiced

    log_trans = build_log_trans(n_bins, sigma_bins, voice_change_penalty)
    path = viterbi_decode(log_obs, log_trans)

    pred_voiced = path < n_bins
    pred_bins = np.where(pred_voiced, path, -1)
    return pred_bins, pred_voiced
