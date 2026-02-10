from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from streaming_asr.eval.evaluator import Evaluator, EvalRequest
from streaming_asr.utils.io import mkdir_p, write_json
from streaming_asr.utils.logging import Logger


def _save_checkpoint(path: Path, state: Dict) -> None:
    mkdir_p(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        run_dir: Path,
        logger: Logger,
        lr: float,
        weight_decay: float,
        grad_clip_norm: float,
        amp: bool,
        log_every_steps: int,
        max_steps_per_epoch: Optional[int],
    ):
        self.model = model
        self.device = device
        self.run_dir = run_dir
        self.logger = logger
        self.grad_clip_norm = float(grad_clip_norm)
        self.amp = bool(amp)
        self.log_every_steps = int(log_every_steps)
        self.max_steps_per_epoch = int(max_steps_per_epoch) if max_steps_per_epoch is not None else None

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        self.global_step = 0

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        feats = batch["feats"].to(self.device)
        feat_lens = batch["feat_lens"].to(self.device)
        labels = batch["labels"].to(self.device)
        label_lens = batch["label_lens"].to(self.device)

        self.opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.amp):
            log_probs, out_lens = self.model(feats, feat_lens)
            # CTCLoss expects (T,B,C), targets (sumL), lengths (B)
            loss = self.ctc(log_probs, labels, out_lens, label_lens)

        self.scaler.scale(loss).backward()
        if self.grad_clip_norm > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        return float(loss.detach().cpu().item())

    def fit(
        self,
        *,
        train_loader: DataLoader,
        dev_loader: Optional[DataLoader],
        dev_tokenizer,
        decoding_kind: str,
        beam_size: int,
        streaming_chunk_size_s: Optional[float],
        epochs: int,
        extra_ckpt_state: Dict,
    ) -> Dict[str, object]:
        self.model.to(self.device)
        best_dev_wer = float("inf")
        best_path = self.run_dir / "checkpoints" / "best.pt"

        history = {"train_loss": [], "dev_wer": []}

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            t0 = time.perf_counter()
            losses = []

            for b, batch in enumerate(train_loader):
                loss = self._train_step(batch)
                losses.append(loss)
                self.global_step += 1

                if (self.global_step % self.log_every_steps) == 0:
                    self.logger.log(f"epoch={epoch} step={self.global_step} loss={loss:.4f}")

                if self.max_steps_per_epoch is not None and (b + 1) >= self.max_steps_per_epoch:
                    break

            mean_loss = float(sum(losses) / max(1, len(losses)))
            history["train_loss"].append({"epoch": epoch, "loss": mean_loss})
            self.logger.log(f"epoch={epoch} train_loss={mean_loss:.4f} time_s={time.perf_counter()-t0:.1f}")

            # Dev eval
            dev_wer = None
            if dev_loader is not None:
                self.model.eval()
                evaluator = Evaluator()
                req = EvalRequest(
                    model=self.model,
                    loader=dev_loader,
                    tokenizer=dev_tokenizer,
                    device=self.device,
                    decoding_kind=str(decoding_kind),
                    beam_size=int(beam_size),
                    streaming_chunk_size_s=streaming_chunk_size_s,
                )
                dev_res = evaluator.evaluate(req)
                dev_wer = float(dev_res["metrics"]["wer"])
                history["dev_wer"].append({"epoch": epoch, "wer": dev_wer})
                self.logger.log(f"epoch={epoch} dev_wer={dev_wer:.4f}")

            # Save last
            last_path = self.run_dir / "checkpoints" / "last.pt"
            state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state": self.model.state_dict(),
                "opt_state": self.opt.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                **extra_ckpt_state,
            }
            _save_checkpoint(last_path, state)

            # Save best
            if dev_wer is not None and dev_wer < best_dev_wer:
                best_dev_wer = dev_wer
                _save_checkpoint(best_path, state)

        write_json(self.run_dir / "history.json", history)
        return {"best_dev_wer": best_dev_wer, "history": history}
