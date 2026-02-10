from __future__ import annotations

import argparse
from pathlib import Path

from streaming_asr.utils.config import Config, parse_config


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, type=str, help="Path to YAML config")
    return p


def load_config_from_args(args: argparse.Namespace) -> Config:
    return parse_config(Path(args.config))

