from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from streaming_asr.decoding.ctc_beam import ctc_prefix_beam_search_ids
from streaming_asr.decoding.ctc_greedy import ctc_greedy_decode_ids
from streaming_asr.data.tokenizer import CharTokenizer
from streaming_asr.utils.metrics import WerStats, aggregate_wer, wer_stats


@dataclass(frozen=True)
class EvalRequest:
    model: torch.nn.Module
    loader: DataLoader
    tokenizer: CharTokenizer
    device: torch.device
    decoding_kind: str  # greedy | beam
    beam_size: int
    streaming_chunk_size_s: Optional[float] = None


class Evaluator:
    def evaluate(self, req: EvalRequest) -> Dict[str, object]:
        model = req.model.to(req.device)
        model.eval()

        total_audio_s = 0.0
        total_time_s = 0.0
        pairs: List[tuple[str, str]] = []
        per_utt: List[dict] = []

        with torch.no_grad():
            for batch in req.loader:
                feats = batch["feats"].to(req.device)
                feat_lens = batch["feat_lens"].to(req.device)
                texts: List[str] = batch["text"]
                utt_ids: List[str] = batch["utt_id"]
                durations = batch["duration_s"].tolist()

                if req.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                log_probs, out_lens = model(feats, feat_lens)  # (T,B,V), (B,)

                if req.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                # Decode
                hyps: List[str] = []
                if req.decoding_kind == "greedy":
                    ids_batch = ctc_greedy_decode_ids(log_probs, blank_id=req.tokenizer.blank_id)
                    hyps = [req.tokenizer.decode_ids(ids) for ids in ids_batch]
                elif req.decoding_kind == "beam":
                    beam = int(req.beam_size)
                    T, B, _V = log_probs.shape
                    for b in range(B):
                        L = int(out_lens[b].item())
                        ids = ctc_prefix_beam_search_ids(log_probs[:L, b, :].cpu(), beam_size=beam, blank_id=req.tokenizer.blank_id)
                        hyps.append(req.tokenizer.decode_ids(ids))
                else:
                    raise ValueError(f"Unknown decoding_kind={req.decoding_kind}")

                if req.device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                batch_time = (t2 - t0)
                total_time_s += batch_time

                for utt_id, ref, hyp, dur in zip(utt_ids, texts, hyps, durations):
                    total_audio_s += float(dur)
                    pairs.append((ref, hyp))
                    st = wer_stats(ref, hyp)
                    per_utt.append(
                        {
                            "utt_id": utt_id,
                            "ref": ref,
                            "hyp": hyp,
                            "num_err": st.num_err,
                            "num_words": st.num_words,
                            "wer": st.wer,
                            "duration_s": float(dur),
                        }
                    )

        agg = aggregate_wer(pairs)
        rtf = (total_time_s / total_audio_s) if total_audio_s > 0 else 0.0
        algo_latency = float(req.streaming_chunk_size_s) if req.streaming_chunk_size_s is not None else 0.0

        return {
            "metrics": {
                "wer": agg.wer,
                "num_err": agg.num_err,
                "num_words": agg.num_words,
                "rtf": float(rtf),
                "total_audio_s": float(total_audio_s),
                "total_time_s": float(total_time_s),
                "algorithmic_latency_s": algo_latency,
            },
            "per_utt": per_utt,
        }

