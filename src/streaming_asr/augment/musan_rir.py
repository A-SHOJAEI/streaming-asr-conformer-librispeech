from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
import torch.nn.functional as F


@dataclass(frozen=True)
class MusanRirConfig:
    apply_prob: float = 0.8
    snr_db_min: float = 5.0
    snr_db_max: float = 20.0
    rir_prob: float = 0.5


def _load_mono(path: str | Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt().clamp_min(1e-8)


def _mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    y = clean + scaled_noise
    y = y / y.abs().max().clamp_min(1.0)
    return y


def _crop_or_pad(x: torch.Tensor, n: int, rng: random.Random) -> torch.Tensor:
    if x.numel() == n:
        return x
    if x.numel() > n:
        start = rng.randint(0, x.numel() - n)
        return x[start : start + n]
    # pad with wrap (more realistic than zeros for noise).
    reps = int(math.ceil(n / x.numel()))
    y = x.repeat(reps)[:n]
    return y


def _convolve_rir(clean: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    # Normalize RIR energy.
    rir = rir / rir.abs().sum().clamp_min(1e-8)
    x = clean.view(1, 1, -1)
    h = rir.view(1, 1, -1)
    y = F.conv1d(x, h, padding=h.size(-1) - 1).view(-1)
    y = y[: clean.numel()]
    y = y / y.abs().max().clamp_min(1.0)
    return y


class MusanRirAugment:
    def __init__(
        self,
        *,
        sample_rate: int,
        noise_paths: List[str],
        rir_paths: List[str],
        cfg: MusanRirConfig = MusanRirConfig(),
    ):
        self.sample_rate = int(sample_rate)
        self.noise_paths = list(noise_paths)
        self.rir_paths = list(rir_paths)
        self.cfg = cfg

    def __call__(self, clean: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if rng.random() > float(self.cfg.apply_prob):
            return clean

        y = clean
        if self.rir_paths and (rng.random() < float(self.cfg.rir_prob)):
            rir = _load_mono(rng.choice(self.rir_paths), self.sample_rate)
            y = _convolve_rir(y, rir)

        if self.noise_paths:
            noise = _load_mono(rng.choice(self.noise_paths), self.sample_rate)
            noise = _crop_or_pad(noise, y.numel(), rng)
            snr = rng.uniform(float(self.cfg.snr_db_min), float(self.cfg.snr_db_max))
            y = _mix_at_snr(y, noise, snr)

        return y

