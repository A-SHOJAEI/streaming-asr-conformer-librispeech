from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from streaming_asr.cli.common import common_parser, load_config_from_args
from streaming_asr.utils.io import mkdir_p, read_json, atomic_write_text


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    p = common_parser("Generate artifacts/report.md from artifacts/results.json.")
    args = p.parse_args()
    cfg = load_config_from_args(args)

    results_path = Path(cfg.artifacts.results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json. Run: make eval (path={results_path})")
    res: Dict[str, Any] = read_json(results_path)

    baseline = res.get("baseline_run")
    runs = res.get("runs", {})
    comps = res.get("comparisons_vs_baseline", {})

    # Determine split names from first run.
    split_names = []
    for r in runs.values():
        split_names = list(r.get("splits", {}).keys())
        break

    lines = []
    lines.append("# Experiment Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{res.get('generated_at_utc', '')}`")
    lines.append(f"- Config: `{res.get('config_path', '')}`")
    lines.append(f"- Baseline run: `{baseline}`")
    lines.append("")

    for split in split_names:
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append("| run | decoding | WER (clean) | RTF | alg. latency (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for run_name, r in runs.items():
            dec = r.get("run_spec", {}).get("decoding", {})
            dec_kind = dec.get("kind", "")
            if dec_kind == "beam":
                dec_str = f"beam{dec.get('beam_size', '')}"
            else:
                dec_str = str(dec_kind or "")
            clean = r["splits"][split]["clean"]["metrics"]
            # decoding kind is not stored explicitly; infer from presence of run_dir's eval? keep blank
            lines.append(
                f"| `{run_name}` | `{dec_str}` | {_fmt(float(clean['wer']))} | {_fmt(float(clean['rtf']))} | {_fmt(float(clean.get('algorithmic_latency_s', 0.0)))} |"
            )
        lines.append("")

        if baseline and baseline in runs:
            lines.append(f"### Paired Deltas vs Baseline (`{baseline}`)")
            lines.append("")
            lines.append("| run | mean(utt WER delta) | 95% bootstrap CI | n_utts |")
            lines.append("|---|---:|---:|---:|")
            for run_name in runs.keys():
                if run_name == baseline:
                    continue
                c = comps.get(run_name, {}).get(split, None)
                if not c:
                    continue
                ci = c.get("paired_utt_wer_delta_ci95", [0.0, 0.0])
                lines.append(
                    f"| `{run_name}` | {_fmt(float(c.get('paired_utt_wer_delta_mean', 0.0)))} | [{_fmt(float(ci[0]))}, {_fmt(float(ci[1]))}] | {int(c.get('n_utts', 0))} |"
                )
            lines.append("")

    # Call out the plan-required baseline and streaming ablation.
    lines.append("## Notes")
    lines.append("")
    lines.append("- Baseline: DeepSpeech2-style Conv + BiLSTM + CTC with greedy decoding.")
    lines.append("- Ablation (streaming constraint): full-context Conformer-CTC vs chunkwise Conformer-CTC (chunk size + limited left context).")
    lines.append("")

    out_path = Path(cfg.artifacts.report_path)
    mkdir_p(out_path.parent)
    atomic_write_text(out_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
