"""Speech dataset and collation for CTC training / evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from streaming_asr.augment.specaugment import SpecAugmentConfig, apply_specaugment
from streaming_asr.data.features import FeatureConfig, extract_log_mel
from streaming_asr.data.manifests import Utterance
from streaming_asr.data.tokenizer import CharTokenizer


@dataclass(frozen=True)
class AugmentFlags:
    specaugment: bool = False
    musan_rir: bool = False


class SpeechDataset(Dataset):
    """Dataset that loads audio, extracts log-mel features, and tokenizes text."""

    def __init__(
        self,
        utts: List[Utterance],
        tokenizer: CharTokenizer,
        feat_cfg: FeatureConfig,
        seed: int = 0,
        augment: Optional[AugmentFlags] = None,
        manifests_dir: Optional[str] = None,
        deterministic_augment: bool = True,
    ) -> None:
        self.utts = utts
        self.tokenizer = tokenizer
        self.feat_cfg = feat_cfg
        self.augment = augment or AugmentFlags()
        self.manifests_dir = manifests_dir
        self.deterministic_augment = deterministic_augment
        self.seed = seed
        self._specaug_cfg = SpecAugmentConfig()

    def __len__(self) -> int:
        return len(self.utts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt = self.utts[idx]

        audio, sr = sf.read(utt.audio_path, dtype="float32")
        wav = torch.from_numpy(audio).float()

        feats = extract_log_mel(wav, self.feat_cfg)

        if self.augment.specaugment:
            if self.deterministic_augment:
                rng = random.Random(self.seed + idx)
            else:
                rng = random.Random()
            feats = apply_specaugment(feats, self._specaug_cfg, rng)

        token_ids = self.tokenizer.encode(utt.text)
        labels = torch.tensor(token_ids, dtype=torch.long)

        return {
            "feats": feats,                            # (T, n_mels)
            "feat_len": feats.shape[0],
            "labels": labels,                          # (L,)
            "label_len": labels.shape[0],
            "text": utt.text,
            "utt_id": utt.utt_id,
            "duration_s": utt.duration_s,
        }


def collate_ctc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dataset samples into a padded batch.

    Samples are sorted by descending feature length so that
    pack_padded_sequence with enforce_sorted=True works correctly
    (required by the DeepSpeech2 model).

    Returns keys expected by both Trainer and Evaluator:
      - feats:      (B, T_max, n_mels)
      - feat_lens:  (B,)
      - labels:     (sum_L,)  concatenated labels (CTC format)
      - label_lens: (B,)
      - text:       list[str]
      - utt_id:     list[str]
      - duration_s: (B,) tensor
    """
    # Sort by descending feature length for pack_padded_sequence
    batch = sorted(batch, key=lambda b: b["feat_len"], reverse=True)

    feats = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    feat_lens = torch.tensor([b["feat_len"] for b in batch], dtype=torch.long)
    label_lens = torch.tensor([b["label_len"] for b in batch], dtype=torch.long)

    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_cat = torch.cat(labels_list, dim=0)

    duration_s = torch.tensor([b["duration_s"] for b in batch], dtype=torch.float32)

    return {
        "feats": feats_padded,
        "feat_lens": feat_lens,
        "labels": labels_cat,
        "label_lens": label_lens,
        "text": [b["text"] for b in batch],
        "utt_id": [b["utt_id"] for b in batch],
        "duration_s": duration_s,
    }
