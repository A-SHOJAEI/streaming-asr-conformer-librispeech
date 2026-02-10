from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DeepSpeech2Config:
    n_mels: int
    vocab_size: int
    n_conv: int = 2
    lstm_hidden: int = 768
    lstm_layers: int = 5
    dropout: float = 0.1


class DeepSpeech2CTC(nn.Module):
    """
    DeepSpeech2-style:
      - 2D conv subsampling front-end
      - BiLSTM encoder
      - Linear CTC head
    """

    def __init__(self, cfg: DeepSpeech2Config):
        super().__init__()
        self.cfg = cfg

        conv = []
        in_ch = 1
        ch = 32
        # Typical DS2 uses two convs with stride 2 in time and freq.
        for i in range(int(cfg.n_conv)):
            conv.append(
                nn.Conv2d(
                    in_ch,
                    ch,
                    kernel_size=(11 if i == 0 else 11, 41 if i == 0 else 21),
                    stride=(2, 2),
                    padding=(5, 20) if i == 0 else (5, 10),
                    bias=False,
                )
            )
            conv.append(nn.BatchNorm2d(ch))
            conv.append(nn.Hardtanh(0.0, 20.0, inplace=True))
            in_ch = ch
            ch = min(ch * 2, 128)
        self.conv = nn.Sequential(*conv)

        # Determine LSTM input dim after convs: (C * F')
        dummy = torch.zeros((1, 1, 100, cfg.n_mels), dtype=torch.float32)
        with torch.no_grad():
            y = self.conv(dummy)
        _, c_out, t_out, f_out = y.shape
        self.subsample_factor = 100 // t_out if t_out > 0 else 4
        lstm_in = int(c_out * f_out)

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=int(cfg.lstm_hidden),
            num_layers=int(cfg.lstm_layers),
            dropout=float(cfg.dropout) if int(cfg.lstm_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(2 * int(cfg.lstm_hidden), int(cfg.vocab_size))

    def _conv_out_lens(self, feat_lens: torch.Tensor) -> torch.Tensor:
        # Each conv uses stride 2 in time.
        L = feat_lens
        for _ in range(int(self.cfg.n_conv)):
            L = torch.div(L + 1, 2, rounding_mode="floor")
        return L.clamp_min(1)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          feats: (B, T, F)
          feat_lens: (B,)
        Returns:
          log_probs: (T', B, vocab)
          out_lens: (B,)
        """
        x = feats.unsqueeze(1)  # (B,1,T,F)
        x = self.conv(x)  # (B,C,T',F')
        B, C, T, Freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * Freq)  # (B,T',C*F')

        out_lens = self._conv_out_lens(feat_lens)
        x = nn.utils.rnn.pack_padded_sequence(x, out_lens.cpu(), batch_first=True, enforce_sorted=True)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        logits = self.proj(x)  # (B,T',V)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()  # (T',B,V)
        return log_probs, out_lens

