"""Character-level tokenizer for CTC-based ASR."""

from __future__ import annotations


class CharTokenizer:
    """Maps characters to integer ids (0 = CTC blank)."""

    def __init__(self, vocab: str) -> None:
        self._vocab = vocab
        # blank_id is 0; characters start at 1
        self._c2i: dict[str, int] = {c: i + 1 for i, c in enumerate(vocab)}
        self._i2c: dict[int, str] = {i + 1: c for i, c in enumerate(vocab)}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def vocab(self) -> str:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + 1  # +1 for blank

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode_ids(self, ids: list[int]) -> str:
        return "".join(self._i2c.get(i, "") for i in ids)
