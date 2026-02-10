from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from streaming_asr.cli.common import common_parser, load_config_from_args
from streaming_asr.data.dataset import AugmentFlags, SpeechDataset, collate_ctc
from streaming_asr.data.features import FeatureConfig
from streaming_asr.data.manifests import load_manifest
from streaming_asr.data.splits import resolve_manifests
from streaming_asr.data.tokenizer import CharTokenizer
from streaming_asr.eval.evaluator import Evaluator, EvalRequest
from streaming_asr.models.chunkwise import seconds_to_frames
from streaming_asr.models.factory import build_conformer, build_deepspeech2
from streaming_asr.utils.io import mkdir_p, read_json, write_json
from streaming_asr.utils.metrics import bootstrap_ci
from streaming_asr.utils.repro import ReproSettings, set_reproducibility


def _device(s: str) -> torch.device:
    s = str(s)
    if s.startswith("cuda") and (not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device(s)


def _resolve_latest_run_dir(run_root: str | Path, exp_name: str) -> Path:
    exp_dir = Path(run_root) / exp_name
    latest = exp_dir / "latest"
    if latest.exists():
        # latest is a symlink to a directory name (relative).
        try:
            return (exp_dir / latest.readlink()).resolve()
        except Exception:
            return latest.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"No runs found for experiment: {exp_name}")
    candidates = [p for p in exp_dir.iterdir() if p.is_dir() and p.name != "latest"]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run dirs found in: {exp_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_checkpoint(run_dir: Path) -> Dict:
    ckpt_dir = run_dir / "checkpoints"
    best = ckpt_dir / "best.pt"
    last = ckpt_dir / "last.pt"
    path = best if best.exists() else last
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for run: {run_dir}")
    return torch.load(path, map_location="cpu")


def _build_model_from_run(cfg, run, tok: CharTokenizer, hop_s: float) -> Tuple[torch.nn.Module, Optional[float]]:
    streaming_chunk_size_s = None
    if run.model.kind == "deepspeech2":
        ds2 = run.model.deepspeech2
        assert ds2 is not None
        bundle = build_deepspeech2(
            n_mels=cfg.features.n_mels,
            vocab_size=tok.vocab_size,
            n_conv=ds2.n_conv,
            lstm_hidden=ds2.lstm_hidden,
            lstm_layers=ds2.lstm_layers,
        )
        model = bundle.model
    elif run.model.kind == "conformer":
        conf = run.model.conformer
        assert conf is not None
        streaming_mode = "full"
        chunk_frames = None
        left_chunks = None
        if run.model.streaming is not None:
            streaming_mode = run.model.streaming.mode
            if streaming_mode == "chunkwise":
                streaming_chunk_size_s = float(run.model.streaming.chunk_size_s or 0.0)
                left_chunks = int(run.model.streaming.left_context_chunks or 0)
                chunk_frames = seconds_to_frames(streaming_chunk_size_s, hop_s=hop_s, subsample_factor=4)
        bundle = build_conformer(
            n_mels=cfg.features.n_mels,
            vocab_size=tok.vocab_size,
            d_model=conf.d_model,
            n_heads=conf.n_heads,
            n_layers=conf.n_layers,
            ff_mult=conf.ff_mult,
            conv_kernel=conf.conv_kernel,
            dropout=conf.dropout,
            streaming_mode=streaming_mode,
            chunk_frames=chunk_frames,
            left_context_chunks=left_chunks,
        )
        model = bundle.model
    else:
        raise ValueError(f"Unknown model.kind={run.model.kind}")

    assert isinstance(model, torch.nn.Module)
    return model, streaming_chunk_size_s


def main() -> None:
    p = common_parser("Evaluate trained runs and write artifacts/results.json.")
    args = p.parse_args()
    cfg = load_config_from_args(args)

    set_reproducibility(ReproSettings(seed=cfg.project.seed, deterministic=True))
    device = _device(cfg.project.device)

    split_paths = resolve_manifests(cfg.data.kind, cfg.data.manifests_dir)
    tok = CharTokenizer(vocab=cfg.tokenizer.vocab)
    feat_cfg = FeatureConfig(
        sample_rate=cfg.features.sample_rate,
        n_mels=cfg.features.n_mels,
        win_length_ms=cfg.features.win_length_ms,
        hop_length_ms=cfg.features.hop_length_ms,
    )
    hop_s = float(cfg.features.hop_length_ms) / 1000.0

    evaluator = Evaluator()

    def make_loader(utts, *, seed: int, aug: AugmentFlags) -> DataLoader:
        ds = SpeechDataset(
            utts=utts,
            tokenizer=tok,
            feat_cfg=feat_cfg,
            seed=seed,
            augment=aug,
            manifests_dir=cfg.data.manifests_dir,
            deterministic_augment=True,
        )
        return DataLoader(
            ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_ctc,
        )

    test_sets: Dict[str, Path] = {}
    if cfg.data.kind == "synthetic":
        assert split_paths.test is not None
        test_sets["test"] = split_paths.test
    else:
        assert split_paths.test_clean is not None and split_paths.test_other is not None
        test_sets["test-clean"] = split_paths.test_clean
        test_sets["test-other"] = split_paths.test_other

    baseline_name = None
    for r in cfg.experiments.runs:
        if r.model.kind == "deepspeech2":
            baseline_name = r.name
            break
    baseline_name = baseline_name or cfg.experiments.runs[0].name

    per_run_results: Dict[str, object] = {}
    baseline_per_utt: Dict[str, Dict[str, float]] = {}  # split -> utt_id -> wer

    for run in cfg.experiments.runs:
        run_dir = _resolve_latest_run_dir(cfg.project.run_root, run.name)
        ckpt = _load_checkpoint(run_dir)

        model, streaming_chunk_size_s = _build_model_from_run(cfg, run, tok, hop_s)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)

        dec_kind = run.decoding.kind
        beam_size = int(run.decoding.beam_size or cfg.eval.beam_size or 1) if dec_kind == "beam" else 1

        run_out: Dict[str, object] = {
            "run_dir": str(run_dir),
            "run_spec": {
                "model": asdict(run.model),
                "augmentation": asdict(run.augmentation),
                "decoding": asdict(run.decoding),
            },
            "splits": {},
        }

        for split_name, manifest_path in test_sets.items():
            utts = load_manifest(manifest_path)
            loader_clean = make_loader(utts, seed=cfg.project.seed, aug=AugmentFlags(specaugment=False, musan_rir=False))
            req = EvalRequest(
                model=model,
                loader=loader_clean,
                tokenizer=tok,
                device=device,
                decoding_kind=dec_kind,
                beam_size=beam_size,
                streaming_chunk_size_s=streaming_chunk_size_s,
            )
            res_clean = evaluator.evaluate(req)
            run_out["splits"][split_name] = {"clean": res_clean}

            if cfg.eval.noisy_test and cfg.data.kind == "openslr":
                loader_noisy = make_loader(
                    utts,
                    seed=int(cfg.eval.noisy_seed),
                    aug=AugmentFlags(specaugment=False, musan_rir=True),
                )
                req_noisy = EvalRequest(
                    model=model,
                    loader=loader_noisy,
                    tokenizer=tok,
                    device=device,
                    decoding_kind=dec_kind,
                    beam_size=beam_size,
                    streaming_chunk_size_s=streaming_chunk_size_s,
                )
                res_noisy = evaluator.evaluate(req_noisy)
                run_out["splits"][split_name]["noisy_fixed_seed"] = res_noisy

        # Persist per-run eval
        write_json(run_dir / "eval.json", run_out)
        per_run_results[run.name] = run_out

        # Save baseline per-utt WER for paired bootstrap deltas.
        if run.name == baseline_name:
            for split_name in test_sets.keys():
                clean = run_out["splits"][split_name]["clean"]
                m = {}
                for r0 in clean["per_utt"]:
                    m[r0["utt_id"]] = float(r0["wer"])
                baseline_per_utt[split_name] = m

    # Comparisons vs baseline (paired, utterance-level WER deltas).
    comparisons: Dict[str, object] = {}
    for run in cfg.experiments.runs:
        if run.name == baseline_name:
            continue
        comp = {}
        run_out = per_run_results[run.name]
        for split_name in test_sets.keys():
            base_map = baseline_per_utt.get(split_name, {})
            run_clean = run_out["splits"][split_name]["clean"]
            deltas: List[float] = []
            for r0 in run_clean["per_utt"]:
                utt_id = r0["utt_id"]
                if utt_id in base_map:
                    deltas.append(float(r0["wer"]) - float(base_map[utt_id]))
            lo, hi = bootstrap_ci(deltas, seed=cfg.project.seed)
            comp[split_name] = {
                "paired_utt_wer_delta_mean": float(sum(deltas) / max(1, len(deltas))),
                "paired_utt_wer_delta_ci95": [lo, hi],
                "n_utts": int(len(deltas)),
            }
        comparisons[run.name] = comp

    out = {
        "generated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": str(args.config),
        "baseline_run": baseline_name,
        "runs": per_run_results,
        "comparisons_vs_baseline": comparisons,
    }
    mkdir_p(Path(cfg.artifacts.results_path).parent)
    write_json(cfg.artifacts.results_path, out)


if __name__ == "__main__":
    main()
