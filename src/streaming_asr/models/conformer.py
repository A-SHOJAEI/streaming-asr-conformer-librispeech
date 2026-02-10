from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from streaming_asr.models.chunkwise import chunkwise_attn_mask


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        d_ff = int(d_model * ff_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = self.ln(x).transpose(1, 2)  # (B,D,T)
        y = self.pw1(y)
        y = F.glu(y, dim=1)  # (B,D,T)
        y = self.dw(y)
        y = self.bn(y)
        y = swish(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2)  # (B,T,D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y = self.ln(x)
        y, _ = self.mha(y, y, y, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(y)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_mult, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_mult, dropout)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.out_ln(x)


class ConvSubsampling(nn.Module):
    """
    2-layer conv2d subsampling that reduces time by factor 4.
    """

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Project conv feature maps to d_model.
        dummy = torch.zeros((1, 1, 100, n_mels), dtype=torch.float32)
        with torch.no_grad():
            y = self.conv(dummy)
        _, c, t, f = y.shape
        self.out_time_factor = 100 // t if t > 0 else 4
        self.proj = nn.Linear(int(c * f), int(d_model))

    def out_lens(self, feat_lens: torch.Tensor) -> torch.Tensor:
        # Two stride-2 convolutions.
        L = feat_lens
        L = torch.div(L + 1, 2, rounding_mode="floor")
        L = torch.div(L + 1, 2, rounding_mode="floor")
        return L.clamp_min(1)

    def forward(self, x: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,F)
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.conv(x)  # (B,C,T',F')
        B, C, T, Freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * Freq)
        x = self.proj(x)
        return x, self.out_lens(feat_lens)


@dataclass(frozen=True)
class ConformerConfig:
    n_mels: int
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    ff_mult: int = 4
    conv_kernel: int = 31
    dropout: float = 0.1

    # Streaming / chunkwise attention.
    streaming_mode: str = "full"  # full | chunkwise
    chunk_frames: Optional[int] = None
    left_context_chunks: Optional[int] = None


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = ConvSubsampling(n_mels=int(cfg.n_mels), d_model=int(cfg.d_model))
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=int(cfg.d_model),
                    n_heads=int(cfg.n_heads),
                    ff_mult=int(cfg.ff_mult),
                    conv_kernel=int(cfg.conv_kernel),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.n_layers))
            ]
        )
        self.ctc = nn.Linear(int(cfg.d_model), int(cfg.vocab_size))

    @property
    def subsample_factor(self) -> int:
        return 4

    def _key_padding_mask(self, out_lens: torch.Tensor, max_len: int) -> torch.Tensor:
        # True for padding positions (ignored by attention).
        # Shape: (B, T)
        B = out_lens.numel()
        idx = torch.arange(max_len, device=out_lens.device).unsqueeze(0).expand(B, max_len)
        return idx >= out_lens.unsqueeze(1)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          feats: (B,T,F)
          feat_lens: (B,)
        Returns:
          log_probs: (T',B,V)
          out_lens: (B,)
        """
        x, out_lens = self.subsample(feats, feat_lens)  # (B,T',D)
        B, T, _ = x.shape
        kpm = self._key_padding_mask(out_lens, T)

        attn_mask = None
        if self.cfg.streaming_mode == "chunkwise":
            if self.cfg.chunk_frames is None or self.cfg.left_context_chunks is None:
                raise ValueError("chunkwise mode requires chunk_frames and left_context_chunks")
            attn_mask = chunkwise_attn_mask(
                T,
                chunk_frames=int(self.cfg.chunk_frames),
                left_chunks=int(self.cfg.left_context_chunks),
                device=x.device,
            )
        elif self.cfg.streaming_mode != "full":
            raise ValueError(f"Unknown streaming_mode={self.cfg.streaming_mode}")

        for b in self.blocks:
            x = b(x, attn_mask=attn_mask, key_padding_mask=kpm)

        logits = self.ctc(x)  # (B,T',V)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()
        return log_probs, out_lens

