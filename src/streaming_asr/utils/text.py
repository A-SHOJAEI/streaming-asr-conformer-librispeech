from __future__ import annotations

import re

_allowed = re.compile(r"[^a-z' ]+")
_spaces = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """
    LibriSpeech-style normalization for simple char/word WER:
    - lowercase
    - keep a-z, space, apostrophe
    - collapse whitespace
    """
    s = s.lower()
    s = _allowed.sub(" ", s)
    s = _spaces.sub(" ", s).strip()
    return s

