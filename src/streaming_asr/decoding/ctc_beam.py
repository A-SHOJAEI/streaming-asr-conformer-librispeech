from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search_ids(
    log_probs: torch.Tensor,
    *,
    beam_size: int,
    blank_id: int = 0,
) -> List[int]:
    """
    Prefix beam search for a single utterance.
    Args:
      log_probs: (T, V) log-probabilities
    Returns:
      best token id sequence
    """
    T, V = log_probs.shape
    beam_size = int(beam_size)
    blank_id = int(blank_id)

    # Maps prefix -> (p_blank, p_nonblank) in log-space.
    beam: Dict[Tuple[int, ...], Tuple[float, float]] = {tuple(): (0.0, float("-inf"))}

    for t in range(T):
        next_beam: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        lp = log_probs[t]

        # Prune per-frame by top-K tokens.
        topk = min(V, max(beam_size * 2, 8))
        vals, idxs = torch.topk(lp, k=topk)
        idxs_l = idxs.tolist()
        vals_l = vals.tolist()

        for prefix, (p_b, p_nb) in beam.items():
            for k, logp in zip(idxs_l, vals_l):
                k = int(k)
                logp = float(logp)

                if k == blank_id:
                    # prefix stays the same, ends with blank
                    nb = next_beam.get(prefix, (float("-inf"), float("-inf")))
                    next_beam[prefix] = (_log_add(nb[0], _log_add(p_b, p_nb) + logp), nb[1])
                    continue

                end = prefix[-1] if prefix else None
                new_prefix = prefix + (k,)

                nb = next_beam.get(new_prefix, (float("-inf"), float("-inf")))
                if end == k:
                    # If repeating token, can come from blank only.
                    p_new_nb = _log_add(nb[1], p_b + logp)
                else:
                    p_new_nb = _log_add(nb[1], _log_add(p_b, p_nb) + logp)
                next_beam[new_prefix] = (nb[0], p_new_nb)

                # Also allow staying on same prefix when token repeats and coming from nonblank.
                if end == k and prefix:
                    nb2 = next_beam.get(prefix, (float("-inf"), float("-inf")))
                    next_beam[prefix] = (nb2[0], _log_add(nb2[1], p_nb + logp))

        # Prune to beam_size by combined prob.
        items = list(next_beam.items())
        items.sort(key=lambda kv: _log_add(kv[1][0], kv[1][1]), reverse=True)
        beam = dict(items[:beam_size])

    best_prefix = max(beam.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)

