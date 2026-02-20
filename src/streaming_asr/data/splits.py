"""Resolve manifest split paths for different data kinds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SplitPaths:
    train: Path
    dev: Path
    test: Optional[Path] = None
    test_clean: Optional[Path] = None
    test_other: Optional[Path] = None


def resolve_manifests(kind: str, manifests_dir: str | Path) -> SplitPaths:
    """Return split manifest paths for the given data kind."""
    d = Path(manifests_dir)

    if kind == "synthetic":
        return SplitPaths(
            train=d / "train.jsonl",
            dev=d / "dev.jsonl",
            test=d / "test.jsonl",
        )

    if kind == "openslr":
        return SplitPaths(
            train=d / "train.jsonl",
            dev=d / "dev.jsonl",
            test_clean=d / "test-clean.jsonl",
            test_other=d / "test-other.jsonl",
        )

    raise ValueError(f"Unknown data kind: {kind}")
