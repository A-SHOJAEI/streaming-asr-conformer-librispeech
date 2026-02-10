from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from streaming_asr.models.conformer import ConformerCTC, ConformerConfig
from streaming_asr.models.deepspeech2 import DeepSpeech2CTC, DeepSpeech2Config


@dataclass(frozen=True)
class ModelBundle:
    model: object
    subsample_factor: int


def build_deepspeech2(*, n_mels: int, vocab_size: int, n_conv: int, lstm_hidden: int, lstm_layers: int):
    cfg = DeepSpeech2Config(
        n_mels=int(n_mels),
        vocab_size=int(vocab_size),
        n_conv=int(n_conv),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
    )
    m = DeepSpeech2CTC(cfg)
    return ModelBundle(model=m, subsample_factor=getattr(m, "subsample_factor", 4))


def build_conformer(
    *,
    n_mels: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ff_mult: int,
    conv_kernel: int,
    dropout: float,
    streaming_mode: str,
    chunk_frames: Optional[int],
    left_context_chunks: Optional[int],
):
    cfg = ConformerConfig(
        n_mels=int(n_mels),
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_mult=int(ff_mult),
        conv_kernel=int(conv_kernel),
        dropout=float(dropout),
        streaming_mode=str(streaming_mode),
        chunk_frames=int(chunk_frames) if chunk_frames is not None else None,
        left_context_chunks=int(left_context_chunks) if left_context_chunks is not None else None,
    )
    m = ConformerCTC(cfg)
    return ModelBundle(model=m, subsample_factor=m.subsample_factor)

