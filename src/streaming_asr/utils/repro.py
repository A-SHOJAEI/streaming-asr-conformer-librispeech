from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproSettings:
    seed: int
    deterministic: bool


def set_reproducibility(settings: ReproSettings) -> None:
    """
    Best-effort reproducibility controls.

    Notes:
    - Some ops remain nondeterministic on GPU depending on CUDA/cuDNN/PyTorch.
    - Determinism can reduce performance.
    """
    seed = int(settings.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For matmul determinism on CUDA (required by torch.use_deterministic_algorithms(True) in some cases).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if settings.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older builds may not support this API.
            pass
    else:
        torch.backends.cudnn.benchmark = True

