# streaming-asr-conformer-librispeech

Noise-robust streaming ASR with a chunkwise (streaming-constrained) Conformer-CTC encoder on LibriSpeech, with a DeepSpeech2-CTC baseline and optional MUSAN+RIR augmentation.

This repo is intentionally "end-to-end runnable": data prep, training, evaluation, and report generation are implemented locally and wired through `make`.

## Problem Statement

For English read speech, quantify the accuracy/latency trade-off when moving from a full-context Conformer-CTC to a streaming-constrained chunkwise Conformer-CTC, and measure how augmentation (SpecAugment and MUSAN noise + RIR convolution) impacts robustness.

Streaming here is modeled as *chunkwise self-attention* with limited left context and fixed algorithmic latency (chunk size), plus CTC decoding.

## Dataset Provenance

This repository supports two data modes (see `configs/*.yaml` and `src/streaming_asr/cli/data.py`):

- **Synthetic smoke dataset** (default; no downloads): generated waveforms and random short transcripts written under `data/synthetic/` and manifests under `data/manifests/` (`src/streaming_asr/data/synthetic.py`). This is only for pipeline sanity checks.
- **OpenSLR downloads** for real experiments (`src/streaming_asr/data/openslr.py`):
  - LibriSpeech (OpenSLR resource `12`): `train-clean-100`, `dev-clean`, `dev-other`, `test-clean`, `test-other`
  - MUSAN (OpenSLR resource `17`)
  - RIRs/Noises (OpenSLR resource `28`)

The exact URLs used by the downloader are hard-coded in `src/streaming_asr/data/openslr.py`:

```text
https://www.openslr.org/resources/12/train-clean-100.tar.gz
https://www.openslr.org/resources/12/dev-clean.tar.gz
https://www.openslr.org/resources/12/dev-other.tar.gz
https://www.openslr.org/resources/12/test-clean.tar.gz
https://www.openslr.org/resources/12/test-other.tar.gz
https://www.openslr.org/resources/17/musan.tar.gz
https://www.openslr.org/resources/28/rirs_noises.zip
```

Manifests are JSONL with fields like `utt_id`, `audio_path`, `duration_s`, `text` (normalized), and are created from LibriSpeech `*.trans.txt` (`src/streaming_asr/data/openslr.py`).

## Methodology (What’s Implemented)

**Features.** Log-mel filterbanks via `torchaudio.transforms.MelSpectrogram` with per-utterance CMVN (`src/streaming_asr/data/features.py`).

**Tokenizer.** Character CTC tokenizer with `<blank>` id `0` and vocab from config (`src/streaming_asr/data/tokenizer.py`). Text is normalized LibriSpeech-style: lowercase, keep `[a-z' ]`, collapse whitespace (`src/streaming_asr/utils/text.py`).

**Models.**

- Baseline: DeepSpeech2-style Conv2D subsampling + BiLSTM + CTC head (`src/streaming_asr/models/deepspeech2.py`).
- Main: Conformer encoder with 2-layer Conv2D subsampling (time reduction factor `4`), Conformer blocks (FFN, MHSA, depthwise conv), and a CTC head (`src/streaming_asr/models/conformer.py`).

**Streaming constraint.** Chunkwise attention is enforced by an attention mask (`src/streaming_asr/models/chunkwise.py`):

- The utterance is still run in one forward pass, but attention is restricted so time `t` can only attend to keys in chunks `[c-left_context_chunks, c]`.
- Within the current chunk, attention is allowed to see "future" frames inside the chunk, which makes the *algorithmic latency* equal to the chunk size.

Chunk size in seconds is converted to post-subsampling frames with `seconds_to_frames(..., subsample_factor=4)` (`src/streaming_asr/models/chunkwise.py`).

**Augmentation.**

- SpecAugment on log-mels (freq/time masks) (`src/streaming_asr/augment/specaugment.py`).
- MUSAN noise mixing (SNR 5-20 dB) and optional RIR convolution (prob 0.5), applied with probability 0.8 (`src/streaming_asr/augment/musan_rir.py`). MUSAN/RIR paths are indexed in `data/manifests/augmentation_index.json` during `make data` for OpenSLR.

**Training.** CTC loss (`torch.nn.CTCLoss(blank=0)`) + AdamW + optional AMP (`src/streaming_asr/train/trainer.py`, `src/streaming_asr/cli/train.py`).

**Decoding.** Greedy CTC and CTC prefix beam search (`src/streaming_asr/decoding/ctc_greedy.py`, `src/streaming_asr/decoding/ctc_beam.py`). No language model is integrated.

**Evaluation + reporting.**

- `make eval` writes `artifacts/results.json` with aggregate metrics and per-utterance `{ref,hyp,wer,...}` (`src/streaming_asr/cli/eval.py`, `src/streaming_asr/eval/evaluator.py`).
- `make report` writes `artifacts/report.md` from `artifacts/results.json` (`src/streaming_asr/cli/report.py`).
- RTF is measured as `(total forward+decode wall time) / (total audio seconds)` inside the evaluator loop.
- Paired deltas vs the baseline are computed from per-utterance WER deltas with percentile bootstrap CI (`src/streaming_asr/utils/metrics.py`).

## Baselines and Ablations (Configured)

The default `configs/smoke.yaml` runs three experiments on the synthetic dataset:

- `baseline_deepspeech2_ctc_greedy`: DS2-CTC, greedy
- `conformer_fullcontext_ctc_greedy`: Conformer-CTC, full-context attention, greedy
- `conformer_chunkwise_0p32s_ctc_greedy`: Conformer-CTC, chunkwise attention with `chunk_size_s=0.32`, `left_context_chunks=4`, greedy

The real grid is defined in `configs/openslr_full.yaml` (LibriSpeech + MUSAN + RIRs) and includes:

- Streaming ablation over chunk sizes `0.32s`, `0.64s`, `1.28s`
- Augmentation ablations under chunkwise `0.64s` (none vs SpecAugment vs MUSAN+RIR vs both)
- Decoding ablation (greedy vs CTC prefix beam search with `beam_size=10`)

## Exact Results (From This Repo’s Artifacts)

The committed artifacts in this workspace are from the synthetic smoke config:

- Generated at: `2026-02-20 06:28:45 UTC`
- Config: `configs/smoke.yaml`
- Report: `artifacts/report.md`
- Machine-readable: `artifacts/results.json`

These numbers are **not** meaningful ASR quality numbers (the dataset is synthetic and tiny: `test` has `16` utterances; training runs 1 epoch with `max_steps_per_epoch=10`). They validate that training/eval/reporting and the streaming plumbing work.

**Table (see `artifacts/report.md`, section "Split: test")**

| run | decoding | WER (test) | RTF | alg. latency (s) |
|---|---:|---:|---:|---:|
| `baseline_deepspeech2_ctc_greedy` | `greedy` | 1.000 | 0.002 | 0.000 |
| `conformer_chunkwise_0p32s_ctc_greedy` | `greedy` | 1.000 | 0.002 | 0.320 |
| `conformer_fullcontext_ctc_greedy` | `greedy` | 1.000 | 0.002 | 0.000 |

(Full-precision floats are in `artifacts/results.json`.)

**Paired deltas vs baseline (see `artifacts/report.md`, section "Paired Deltas vs Baseline")**

| run | mean(utt WER delta) | 95% bootstrap CI | n_utts |
|---|---:|---:|---:|
| `conformer_chunkwise_0p32s_ctc_greedy` | 0.000 | [0.000, 0.000] | 16 |
| `conformer_fullcontext_ctc_greedy` | 0.000 | [0.000, 0.000] | 16 |

Per-utterance refs/hyps are stored in `artifacts/results.json` under `runs/*/splits/test/clean/per_utt` (e.g., some hyps are empty strings in the smoke run).

## Reproducibility / How To Run

This repo uses a repo-local virtualenv (`.venv`) and never installs into system Python (see `scripts/bootstrap_venv.sh` and `Makefile`).

### Smoke run (no downloads)

```bash
make all
```

Outputs:

- `artifacts/results.json`
- `artifacts/report.md`
- `runs/<experiment>/<timestamp>/` with `checkpoints/`, `history.json`, `eval.json`, logs

### Real LibriSpeech runs (OpenSLR)

```bash
make data   CONFIG=configs/openslr_full.yaml
make train  CONFIG=configs/openslr_full.yaml
make eval   CONFIG=configs/openslr_full.yaml
make report CONFIG=configs/openslr_full.yaml
```

Notes (based on the actual config/code):

- `configs/openslr_full.yaml` defaults to `project.device: cuda` and `train.amp: true`. If you run on CPU, set `device: cpu` and disable AMP.
- Downloads are stored under `data/downloads/`, extracted under `data/extracted/`, manifests under `data/manifests/`.
- OpenSLR downloads use resumable HTTP + best-effort MD5 scraping from OpenSLR pages (`src/streaming_asr/utils/download.py`).

## Limitations (Current State)

- "Streaming" is enforced by an attention mask but still evaluated in a single full-utterance forward pass; there is no incremental state caching, and measured RTF is not the same as an online chunk-by-chunk system.
- Chunkwise attention allows within-chunk future context; algorithmic latency is modeled as `chunk_size_s`, but this is not strict frame-synchronous streaming.
- No language model integration; beam search is acoustic-only CTC prefix beam search.
- WER is computed with a simple tokenizer/normalizer and whitespace word-splitting (`src/streaming_asr/utils/metrics.py`, `src/streaming_asr/utils/text.py`); results may differ from standard LibriSpeech scoring toolchains.
- The current committed artifacts are from the synthetic smoke config; they are not LibriSpeech numbers.

## Next Research Steps (Concrete)

1. Implement true online inference: chunk-by-chunk encoder forward with cached keys/values (or a streaming-friendly Conformer variant), and report end-to-end latency (including compute) vs WER.
2. Add strict causal masking within chunk (no within-chunk future) and compare against the current "within-chunk lookahead" formulation.
3. Add LM integration (e.g., n-gram) and evaluate `greedy` vs `beam` trade-offs under streaming constraints.
4. Extend tokenization beyond pure chars (BPE/wordpieces) and compare CTC stability, WER, and decoding cost.
5. Use the built-in `noisy_fixed_seed` eval path in `src/streaming_asr/cli/eval.py` on OpenSLR runs to quantify robustness gains from MUSAN+RIR and tune SNR/RIR sampling.
