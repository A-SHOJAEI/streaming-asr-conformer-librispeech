from __future__ import annotations

from pathlib import Path

from streaming_asr.cli.common import common_parser, load_config_from_args
from streaming_asr.data.openslr import prepare_openslr
from streaming_asr.data.synthetic import prepare_synthetic_manifests
from streaming_asr.utils.io import mkdir_p, write_json


def main() -> None:
    p = common_parser("Prepare data/manifests (synthetic smoke or OpenSLR downloads).")
    args = p.parse_args()
    cfg = load_config_from_args(args)

    root = Path(cfg.data.root)
    mkdir_p(root)
    manifests_dir = Path(cfg.data.manifests_dir)
    mkdir_p(manifests_dir)

    if cfg.data.kind == "synthetic":
        assert cfg.data.synthetic is not None
        out = prepare_synthetic_manifests(
            out_root=root,
            manifests_dir=manifests_dir,
            seed=cfg.project.seed,
            sample_rate=cfg.data.synthetic.sample_rate,
            min_sec=cfg.data.synthetic.min_sec,
            max_sec=cfg.data.synthetic.max_sec,
            vocab=cfg.data.synthetic.vocab,
            train_samples=cfg.data.synthetic.train_samples,
            dev_samples=cfg.data.synthetic.dev_samples,
            test_samples=cfg.data.synthetic.test_samples,
        )
        write_json(manifests_dir / "prepared.json", {"kind": "synthetic", "manifests": {k: str(v) for k, v in out.items()}})
        return

    if cfg.data.kind == "openslr":
        assert cfg.data.openslr is not None
        if not cfg.data.downloads_dir or not cfg.data.extracted_dir:
            raise ValueError("openslr data kind requires data.downloads_dir and data.extracted_dir")
        out = prepare_openslr(
            downloads_dir=cfg.data.downloads_dir,
            extracted_dir=cfg.data.extracted_dir,
            manifests_dir=cfg.data.manifests_dir,
            librispeech_splits=cfg.data.openslr.librispeech_splits,
            include_musan=cfg.data.openslr.include_musan,
            include_rirs=cfg.data.openslr.include_rirs,
        )
        write_json(manifests_dir / "prepared.json", {"kind": "openslr", "manifests": {k: str(v) for k, v in out.items()}})
        return

    raise ValueError(f"Unknown data.kind={cfg.data.kind}")


if __name__ == "__main__":
    main()

