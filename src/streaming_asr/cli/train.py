from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from streaming_asr.cli.common import common_parser, load_config_from_args
from streaming_asr.data.dataset import AugmentFlags, SpeechDataset, collate_ctc
from streaming_asr.data.features import FeatureConfig
from streaming_asr.data.manifests import load_manifest
from streaming_asr.data.splits import resolve_manifests
from streaming_asr.data.tokenizer import CharTokenizer
from streaming_asr.models.chunkwise import seconds_to_frames
from streaming_asr.models.factory import build_conformer, build_deepspeech2
from streaming_asr.train.trainer import Trainer
from streaming_asr.utils.io import mkdir_p, write_json
from streaming_asr.utils.logging import Logger
from streaming_asr.utils.repro import ReproSettings, set_reproducibility
from streaming_asr.utils.run import make_run_dir


def _device(s: str) -> torch.device:
    s = str(s)
    if s.startswith("cuda") and (not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device(s)


def main() -> None:
    p = common_parser("Train baseline and ablations defined in config.")
    args = p.parse_args()
    cfg = load_config_from_args(args)

    set_reproducibility(ReproSettings(seed=cfg.project.seed, deterministic=cfg.project.deterministic))
    device = _device(cfg.project.device)

    split_paths = resolve_manifests(cfg.data.kind, cfg.data.manifests_dir)
    if not split_paths.train.exists() or not split_paths.dev.exists():
        raise FileNotFoundError("Manifests not found. Run: make data")

    feat_cfg = FeatureConfig(
        sample_rate=cfg.features.sample_rate,
        n_mels=cfg.features.n_mels,
        win_length_ms=cfg.features.win_length_ms,
        hop_length_ms=cfg.features.hop_length_ms,
    )
    tok = CharTokenizer(vocab=cfg.tokenizer.vocab)

    train_utts = load_manifest(split_paths.train)
    dev_utts = load_manifest(split_paths.dev)

    hop_s = float(cfg.features.hop_length_ms) / 1000.0

    for run in cfg.experiments.runs:
        run_dir = make_run_dir(cfg.project.run_root, run.name)
        logger = Logger(path=run_dir / "train.log")
        logger.log(f"run={run.name} device={device} seed={cfg.project.seed} deterministic={cfg.project.deterministic}")

        write_json(run_dir / "config.json", {"config_path": str(args.config), "run": run.name, "cfg": asdict(cfg)})

        aug_train = AugmentFlags(specaugment=run.augmentation.specaugment, musan_rir=run.augmentation.musan_rir)
        aug_dev = AugmentFlags(specaugment=False, musan_rir=False)

        train_ds = SpeechDataset(
            utts=train_utts,
            tokenizer=tok,
            feat_cfg=feat_cfg,
            seed=cfg.project.seed,
            augment=aug_train,
            manifests_dir=cfg.data.manifests_dir,
            deterministic_augment=True,
        )
        dev_ds = SpeechDataset(
            utts=dev_utts,
            tokenizer=tok,
            feat_cfg=feat_cfg,
            seed=cfg.project.seed,
            augment=aug_dev,
            manifests_dir=cfg.data.manifests_dir,
            deterministic_augment=True,
        )

        g = torch.Generator()
        g.manual_seed(int(cfg.project.seed))

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_ctc,
            generator=g,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_ctc,
        )

        streaming_chunk_size_s = None
        if run.model.kind == "deepspeech2":
            assert run.model.deepspeech2 is not None
            bundle = build_deepspeech2(
                n_mels=cfg.features.n_mels,
                vocab_size=tok.vocab_size,
                n_conv=run.model.deepspeech2.n_conv,
                lstm_hidden=run.model.deepspeech2.lstm_hidden,
                lstm_layers=run.model.deepspeech2.lstm_layers,
            )
        elif run.model.kind == "conformer":
            assert run.model.conformer is not None
            streaming_mode = "full"
            chunk_frames = None
            left_chunks = None
            if run.model.streaming is not None:
                streaming_mode = run.model.streaming.mode
                if streaming_mode == "chunkwise":
                    if run.model.streaming.chunk_size_s is None or run.model.streaming.left_context_chunks is None:
                        raise ValueError("chunkwise requires chunk_size_s and left_context_chunks")
                    streaming_chunk_size_s = float(run.model.streaming.chunk_size_s)
                    chunk_frames = seconds_to_frames(streaming_chunk_size_s, hop_s=hop_s, subsample_factor=4)
                    left_chunks = int(run.model.streaming.left_context_chunks)

            bundle = build_conformer(
                n_mels=cfg.features.n_mels,
                vocab_size=tok.vocab_size,
                d_model=run.model.conformer.d_model,
                n_heads=run.model.conformer.n_heads,
                n_layers=run.model.conformer.n_layers,
                ff_mult=run.model.conformer.ff_mult,
                conv_kernel=run.model.conformer.conv_kernel,
                dropout=run.model.conformer.dropout,
                streaming_mode=streaming_mode,
                chunk_frames=chunk_frames,
                left_context_chunks=left_chunks,
            )
        else:
            raise ValueError(f"Unknown model.kind={run.model.kind}")

        model = bundle.model  # type: ignore[assignment]
        assert isinstance(model, torch.nn.Module)

        trainer = Trainer(
            model=model,
            device=device,
            run_dir=run_dir,
            logger=logger,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            grad_clip_norm=cfg.train.grad_clip_norm,
            amp=cfg.train.amp and (device.type == "cuda"),
            log_every_steps=cfg.train.log_every_steps,
            max_steps_per_epoch=cfg.train.max_steps_per_epoch,
        )

        extra_state = {
            "tokenizer": {"kind": "char", "vocab": tok.vocab},
            "model_kind": run.model.kind,
        }
        res = trainer.fit(
            train_loader=train_loader,
            dev_loader=dev_loader,
            dev_tokenizer=tok,
            decoding_kind="greedy",
            beam_size=1,
            streaming_chunk_size_s=streaming_chunk_size_s,
            epochs=cfg.train.epochs,
            extra_ckpt_state=extra_state,
        )

        write_json(run_dir / "train_summary.json", res)


if __name__ == "__main__":
    main()

