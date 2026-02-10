from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    dst_path: Path
    expected_bytes: Optional[int] = None
    expected_md5: Optional[str] = None
    expected_sha256: Optional[str] = None


def _hash_file(path: Path, algo: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _head_content_length(url: str, *, timeout_s: int = 30) -> Optional[int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout_s)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def download_with_resume(spec: DownloadSpec, *, timeout_s: int = 30) -> Path:
    """
    Resumable HTTP download using Range requests when supported.
    """
    spec.dst_path.parent.mkdir(parents=True, exist_ok=True)

    expected_bytes = spec.expected_bytes
    if expected_bytes is None:
        expected_bytes = _head_content_length(spec.url, timeout_s=timeout_s)

    tmp = spec.dst_path.with_suffix(spec.dst_path.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0

    headers = {}
    if have > 0:
        headers["Range"] = f"bytes={have}-"

    with requests.get(spec.url, stream=True, headers=headers, timeout=timeout_s, allow_redirects=True) as r:
        r.raise_for_status()

        # If server ignored Range, restart from scratch.
        if have > 0 and r.status_code == 200:
            tmp.unlink(missing_ok=True)
            have = 0

        total = expected_bytes
        if total is not None and have > 0:
            total = max(0, total - have)

        bar = tqdm(total=total, unit="B", unit_scale=True, desc=spec.dst_path.name)
        bar.update(0)
        mode = "ab" if have > 0 else "wb"
        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()

    tmp.replace(spec.dst_path)
    return spec.dst_path


def verify_file(
    path: Path,
    *,
    expected_bytes: Optional[int] = None,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> None:
    if expected_bytes is not None:
        size = path.stat().st_size
        if int(size) != int(expected_bytes):
            raise ValueError(f"Size mismatch for {path}: got {size} bytes, expected {expected_bytes}")

    if expected_md5 is not None:
        got = _hash_file(path, "md5")
        if got.lower() != expected_md5.lower():
            raise ValueError(f"MD5 mismatch for {path}: got {got}, expected {expected_md5}")

    if expected_sha256 is not None:
        got = _hash_file(path, "sha256")
        if got.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {path}: got {got}, expected {expected_sha256}")


def openslr_md5_for(url: str, *, timeout_s: int = 30) -> Optional[str]:
    """
    Best-effort MD5 discovery by scraping the relevant OpenSLR page.
    Example:
      https://www.openslr.org/resources/12/train-clean-100.tar.gz  ->  https://www.openslr.org/12/
    """
    m = re.search(r"openslr\\.org/resources/(\\d+)/(.*)$", url)
    if not m:
        return None
    rid = m.group(1)
    filename = m.group(2)
    page = f"https://www.openslr.org/{rid}/"

    try:
        r = requests.get(page, timeout=timeout_s)
        r.raise_for_status()
        html = r.text
    except Exception:
        return None

    # Find a window around the filename; then grab the first 32-hex MD5 nearby.
    idx = html.find(filename)
    if idx == -1:
        return None
    window = html[max(0, idx - 500) : idx + 500]
    m2 = re.search(r"([a-fA-F0-9]{32})", window)
    return m2.group(1).lower() if m2 else None


def download_and_verify(
    url: str,
    dst_path: Path,
    *,
    expected_bytes: Optional[int] = None,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    allow_openslr_md5_scrape: bool = True,
) -> Path:
    if expected_md5 is None and allow_openslr_md5_scrape:
        expected_md5 = openslr_md5_for(url)

    if dst_path.exists():
        try:
            verify_file(dst_path, expected_bytes=expected_bytes, expected_md5=expected_md5, expected_sha256=expected_sha256)
            return dst_path
        except Exception:
            # Keep a backup for debugging.
            bak = dst_path.with_suffix(dst_path.suffix + f".bad.{int(time.time())}")
            dst_path.replace(bak)

    spec = DownloadSpec(url=url, dst_path=dst_path, expected_bytes=expected_bytes, expected_md5=expected_md5, expected_sha256=expected_sha256)
    out = download_with_resume(spec)
    verify_file(out, expected_bytes=expected_bytes, expected_md5=expected_md5, expected_sha256=expected_sha256)
    return out

