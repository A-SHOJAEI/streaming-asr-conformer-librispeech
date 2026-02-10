from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class WerStats:
    num_err: int
    num_words: int

    @property
    def wer(self) -> float:
        return float(self.num_err) / float(self.num_words) if self.num_words > 0 else 0.0


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    # Standard DP Levenshtein distance.
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,  # substitution
            )
            prev = cur
    return dp[m]


def wer_stats(ref: str, hyp: str) -> WerStats:
    r = [w for w in ref.split(" ") if w]
    h = [w for w in hyp.split(" ") if w]
    return WerStats(num_err=_edit_distance(r, h), num_words=len(r))


def aggregate_wer(pairs: Iterable[Tuple[str, str]]) -> WerStats:
    err = 0
    words = 0
    for ref, hyp in pairs:
        st = wer_stats(ref, hyp)
        err += st.num_err
        words += st.num_words
    return WerStats(num_err=err, num_words=words)


def bootstrap_ci(
    deltas: List[float],
    *,
    num_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for a list of paired deltas.
    Returns (lower, upper).
    """
    import random

    rng = random.Random(int(seed))
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(int(num_samples)):
        s = 0.0
        for _i in range(n):
            s += deltas[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo = samples[int((alpha / 2.0) * len(samples))]
    hi = samples[int((1.0 - alpha / 2.0) * len(samples)) - 1]
    return (float(lo), float(hi))

