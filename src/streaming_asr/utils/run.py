from __future__ import annotations

import datetime as _dt
from pathlib import Path

from streaming_asr.utils.io import mkdir_p


def timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def make_run_dir(run_root: str | Path, exp_name: str) -> Path:
    run_root = Path(run_root)
    d = mkdir_p(run_root / exp_name / timestamp())
    latest = run_root / exp_name / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(d.name)  # relative symlink inside exp dir
    except Exception:
        # Symlinks may fail on some filesystems; ignore.
        pass
    return d

