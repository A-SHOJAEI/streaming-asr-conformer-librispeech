from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {path}")
    return data


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    device: str
    deterministic: bool
    run_root: str


@dataclass(frozen=True)
class SyntheticDataConfig:
    train_samples: int
    dev_samples: int
    test_samples: int
    sample_rate: int
    min_sec: float
    max_sec: float
    vocab: str


@dataclass(frozen=True)
class OpenSlrDataConfig:
    librispeech_splits: List[str]
    include_musan: bool
    include_rirs: bool


@dataclass(frozen=True)
class DataConfig:
    kind: str
    root: str
    manifests_dir: str
    downloads_dir: Optional[str] = None
    extracted_dir: Optional[str] = None
    synthetic: Optional[SyntheticDataConfig] = None
    openslr: Optional[OpenSlrDataConfig] = None


@dataclass(frozen=True)
class FeaturesConfig:
    sample_rate: int
    n_mels: int
    win_length_ms: int
    hop_length_ms: int


@dataclass(frozen=True)
class TokenizerConfig:
    kind: str
    vocab: str


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    amp: bool
    log_every_steps: int
    max_steps_per_epoch: Optional[int] = None


@dataclass(frozen=True)
class EvalConfig:
    beam_size: int
    noisy_test: bool
    noisy_seed: int = 2026


@dataclass(frozen=True)
class ArtifactsConfig:
    results_path: str
    report_path: str


@dataclass(frozen=True)
class StreamingConfig:
    mode: str  # full | chunkwise
    chunk_size_s: Optional[float] = None
    left_context_chunks: Optional[int] = None


@dataclass(frozen=True)
class DeepSpeech2ModelConfig:
    n_conv: int
    lstm_hidden: int
    lstm_layers: int


@dataclass(frozen=True)
class ConformerModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    conv_kernel: int
    dropout: float


@dataclass(frozen=True)
class ModelConfig:
    kind: str  # deepspeech2 | conformer
    deepspeech2: Optional[DeepSpeech2ModelConfig] = None
    conformer: Optional[ConformerModelConfig] = None
    streaming: Optional[StreamingConfig] = None


@dataclass(frozen=True)
class AugmentationConfig:
    specaugment: bool
    musan_rir: bool


@dataclass(frozen=True)
class DecodingConfig:
    kind: str  # greedy | beam
    beam_size: Optional[int] = None


@dataclass(frozen=True)
class ExperimentRunConfig:
    name: str
    model: ModelConfig
    augmentation: AugmentationConfig
    decoding: DecodingConfig


@dataclass(frozen=True)
class ExperimentsConfig:
    runs: List[ExperimentRunConfig]


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    tokenizer: TokenizerConfig
    train: TrainConfig
    eval: EvalConfig
    artifacts: ArtifactsConfig
    experiments: ExperimentsConfig


def _dc(cls, d: Dict[str, Any]):
    return cls(**d)


def parse_config(path: str | Path) -> Config:
    raw = load_yaml(path)

    project = _dc(ProjectConfig, _req(raw, "project"))
    features = _dc(FeaturesConfig, _req(raw, "features"))
    tokenizer = _dc(TokenizerConfig, _req(raw, "tokenizer"))
    train = _dc(TrainConfig, _req(raw, "train"))
    eval_cfg = _dc(EvalConfig, _req(raw, "eval"))
    artifacts = _dc(ArtifactsConfig, _req(raw, "artifacts"))

    data_raw = _req(raw, "data")
    data_kind = _req(data_raw, "kind")
    synthetic = None
    openslr = None
    if data_kind == "synthetic":
        synthetic = _dc(SyntheticDataConfig, _req(data_raw, "synthetic"))
    elif data_kind == "openslr":
        openslr = _dc(OpenSlrDataConfig, _req(data_raw, "openslr"))
    else:
        raise ValueError(f"Unknown data.kind={data_kind}")
    data = DataConfig(
        kind=data_kind,
        root=_req(data_raw, "root"),
        manifests_dir=_req(data_raw, "manifests_dir"),
        downloads_dir=data_raw.get("downloads_dir"),
        extracted_dir=data_raw.get("extracted_dir"),
        synthetic=synthetic,
        openslr=openslr,
    )

    runs: List[ExperimentRunConfig] = []
    for r in _req(_req(raw, "experiments"), "runs"):
        model_raw = _req(r, "model")
        kind = _req(model_raw, "kind")

        streaming = None
        if "streaming" in model_raw:
            streaming = _dc(StreamingConfig, _req(model_raw, "streaming"))

        if kind == "deepspeech2":
            ds2 = _dc(DeepSpeech2ModelConfig, _req(model_raw, "deepspeech2"))
            model = ModelConfig(kind=kind, deepspeech2=ds2, streaming=streaming)
        elif kind == "conformer":
            conf = _dc(ConformerModelConfig, _req(model_raw, "conformer"))
            model = ModelConfig(kind=kind, conformer=conf, streaming=streaming)
        else:
            raise ValueError(f"Unknown model.kind={kind}")

        aug = _dc(AugmentationConfig, _req(r, "augmentation"))
        dec = _dc(DecodingConfig, _req(r, "decoding"))
        runs.append(ExperimentRunConfig(name=_req(r, "name"), model=model, augmentation=aug, decoding=dec))

    experiments = ExperimentsConfig(runs=runs)

    return Config(
        project=project,
        data=data,
        features=features,
        tokenizer=tokenizer,
        train=train,
        eval=eval_cfg,
        artifacts=artifacts,
        experiments=experiments,
    )

