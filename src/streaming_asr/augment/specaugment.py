from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecAugmentConfig:
    freq_masks: int = 2
    time_masks: int = 2
    max_freq_mask: int = 10
    max_time_mask_ratio: float = 0.05


def apply_specaugment(feats: torch.Tensor, cfg: SpecAugmentConfig, rng: random.Random) -> torch.Tensor:
    """
    SpecAugment on log-mel features (frames, n_mels).
    """
    x = feats.clone()
    T, F = x.shape

    for _ in range(int(cfg.freq_masks)):
        w = rng.randint(0, min(int(cfg.max_freq_mask), F))
        if w <= 0:
            continue
        f0 = rng.randint(0, max(0, F - w))
        x[:, f0 : f0 + w] = 0.0

    max_time = max(1, int(float(cfg.max_time_mask_ratio) * T))
    for _ in range(int(cfg.time_masks)):
        w = rng.randint(0, min(max_time, T))
        if w <= 0:
            continue
        t0 = rng.randint(0, max(0, T - w))
        x[t0 : t0 + w, :] = 0.0

    return x

