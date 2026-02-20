"""JSONL manifest loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from streaming_asr.utils.io import read_jsonl


@dataclass
class Utterance:
    utt_id: str
    audio_path: str
    text: str
    duration_s: float


def load_manifest(path: str | Path) -> List[Utterance]:
    """Read a JSONL manifest file and return a list of Utterance objects."""
    items: List[Utterance] = []
    for obj in read_jsonl(path):
        items.append(Utterance(
            utt_id=str(obj.get("utt_id", obj.get("id", f"utt_{len(items)}"))),
            audio_path=str(obj["audio_path"]),
            text=str(obj.get("text", "")),
            duration_s=float(obj.get("duration_s", 0.0)),
        ))
    return items
