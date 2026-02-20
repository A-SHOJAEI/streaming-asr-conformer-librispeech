"""Audio feature extraction (log-mel filterbanks)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    win_length_ms: int = 25
    hop_length_ms: int = 10

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_length_ms / 1000)

    @property
    def n_fft(self) -> int:
        return self.win_length


def extract_log_mel(wav: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Extract log-mel features from a 1-D waveform tensor.

    Args:
        wav: (T,) waveform
        cfg: feature config

    Returns:
        (n_frames, n_mels) log-mel features
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    spec = mel_spec(wav)       # (n_mels, n_frames)
    log_mel = amp_to_db(spec)  # (n_mels, n_frames)
    return log_mel.transpose(0, 1)  # (n_frames, n_mels)
