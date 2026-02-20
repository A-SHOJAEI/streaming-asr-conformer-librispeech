"""Generate synthetic (random noise) speech data for smoke tests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import soundfile as sf
import numpy as np

from streaming_asr.utils.io import mkdir_p, write_jsonl


def _random_transcript(vocab: str, rng: random.Random, min_len: int = 2, max_len: int = 12) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(vocab) for _ in range(length)).strip() or "a"


def prepare_synthetic_manifests(
    *,
    out_root: str | Path,
    manifests_dir: str | Path,
    seed: int,
    sample_rate: int,
    min_sec: float,
    max_sec: float,
    vocab: str,
    train_samples: int,
    dev_samples: int,
    test_samples: int,
) -> Dict[str, Path]:
    """Generate synthetic WAV files and JSONL manifests."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    out_root = Path(out_root)
    manifests_dir = Path(manifests_dir)
    mkdir_p(manifests_dir)

    splits = {
        "train": train_samples,
        "dev": dev_samples,
        "test": test_samples,
    }

    results: Dict[str, Path] = {}

    for split_name, n_samples in splits.items():
        wav_dir = out_root / "wavs" / split_name
        mkdir_p(wav_dir)

        rows = []
        for i in range(n_samples):
            duration = rng.uniform(min_sec, max_sec)
            n_frames = int(duration * sample_rate)
            audio = np_rng.randn(n_frames).astype(np.float32) * 0.01

            utt_id = f"{split_name}_{i:05d}"
            wav_path = wav_dir / f"{utt_id}.wav"
            sf.write(str(wav_path), audio, sample_rate)

            text = _random_transcript(vocab, rng)
            actual_duration = n_frames / sample_rate

            rows.append({
                "utt_id": utt_id,
                "audio_path": str(wav_path),
                "text": text,
                "duration_s": round(actual_duration, 4),
            })

        manifest_path = manifests_dir / f"{split_name}.jsonl"
        write_jsonl(manifest_path, rows)
        results[split_name] = manifest_path

    return results
