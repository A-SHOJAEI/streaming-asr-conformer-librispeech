from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ChunkwiseSpec:
    mode: str  # full | chunkwise
    chunk_size_frames: Optional[int] = None
    left_context_chunks: Optional[int] = None


def chunkwise_attn_mask(
    T: int, *, chunk_frames: int, left_chunks: int, device: torch.device
) -> torch.Tensor:
    """
    Returns a float mask suitable for torch.nn.MultiheadAttention(attn_mask=...):
      - shape (T, T)
      - entries are 0 for allowed, -inf for disallowed

    Chunkwise definition:
      - Query at time t in chunk c can attend to:
        keys in chunks [c-left_chunks, c], and any position within chunk c (including "future" within chunk).
    """
    T = int(T)
    chunk_frames = max(1, int(chunk_frames))
    left_chunks = max(0, int(left_chunks))

    # Build boolean "disallow" mask, then convert to -inf.
    disallow = torch.ones((T, T), dtype=torch.bool, device=device)
    for t in range(T):
        c = t // chunk_frames
        k0 = max(0, (c - left_chunks) * chunk_frames)
        k1 = min(T, (c + 1) * chunk_frames)  # exclusive
        disallow[t, k0:k1] = False

    attn_mask = torch.zeros((T, T), dtype=torch.float32, device=device)
    attn_mask.masked_fill_(disallow, float("-inf"))
    return attn_mask


def seconds_to_frames(seconds: float, hop_s: float, subsample_factor: int) -> int:
    """
    Convert chunk size in seconds to frames after subsampling.
    """
    base_frames = int(math.ceil(float(seconds) / float(hop_s)))
    return max(1, int(math.ceil(base_frames / int(subsample_factor))))

