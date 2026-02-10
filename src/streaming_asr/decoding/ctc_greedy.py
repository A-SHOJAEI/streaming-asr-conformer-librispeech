from __future__ import annotations

from typing import List

import torch


def ctc_greedy_decode_ids(log_probs: torch.Tensor, *, blank_id: int = 0) -> List[List[int]]:
    """
    Args:
      log_probs: (T, B, V)
    Returns:
      decoded ids per batch element (blank removed, repeats collapsed)
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected (T,B,V), got {tuple(log_probs.shape)}")
    best = torch.argmax(log_probs, dim=-1)  # (T,B)
    T, B = best.shape
    out: List[List[int]] = []
    for b in range(B):
        seq = best[:, b].tolist()
        collapsed = []
        prev = None
        for t in range(T):
            v = int(seq[t])
            if v == prev:
                continue
            prev = v
            if v == int(blank_id):
                continue
            collapsed.append(v)
        out.append(collapsed)
    return out

