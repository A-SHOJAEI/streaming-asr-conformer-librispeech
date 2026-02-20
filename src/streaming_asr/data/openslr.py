"""OpenSLR / LibriSpeech data preparation (placeholder for smoke test)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def prepare_openslr(
    *,
    downloads_dir: str | Path,
    extracted_dir: str | Path,
    manifests_dir: str | Path,
    librispeech_splits: List[str],
    include_musan: bool,
    include_rirs: bool,
) -> Dict[str, Path]:
    """Download and prepare LibriSpeech (+ optionally MUSAN/RIRs).

    This is a stub for the smoke test; a full implementation would
    download tarballs from openslr.org and build manifests.
    """
    raise NotImplementedError(
        "OpenSLR data preparation requires internet access and large downloads. "
        "Use data.kind=synthetic for smoke tests."
    )
