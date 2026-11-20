"""Stage 1: train native MixtralLM from random weights on healthcare text ONLY.

No external pretrained language-model checkpoint is loaded. The tokenizer is trained
on healthcare documents from Hugging Face; torchrun supports DDP / FSDP2.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mixtral_lab import ModelConfig
from mixtral_lab.data import get_tokenizer
from mixtral_lab.healthcare_corpus import HealthcarePackedTokens, DEFAULT_HF_CORPUS
from mixtral_lab.train_utils import (init_distributed, make_model, lr_schedule,
                           save_checkpoint, resume_checkpoint, MetricsLog)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/moe_1b.json")
    p.add_argument("--tokenizer", default="artifacts/healthcare_tokenizer.json")
    p.add_argument("--source", choices=["hf", "local"], default="hf")
    p.add_argument("--dataset", default=DEFAULT_HF_CORPUS)
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="train")
    p.add_argument("--local-jsonl")
    p.add_argument("--min-clinical-hits", type=int, default=1)
    p.add_argument("--allow-unknown-license", action="store_true")
    p.add_argument("--val-pct", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp2"], default="single")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Force CPU to train models larger than 4GB GPU VRAM")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/healthcare_pretrained")
    p.add_argument("--metrics-log", help="JSON Lines file for per-step metrics (default: <out>/metrics.jsonl)")
    p.add_argument("--resume", help="Resume weights and optimizer, not precise streaming dataset position")
    return p


def loader_for(a, partition, rank, world, device):
    ds = HealthcarePackedTokens(tokenizer_path=a.tokenizer, seq_len=a.seq_len,
        source=a.source, dataset=a.dataset, dataset_config=a.dataset_config,
        split=a.split, local_jsonl=a.local_jsonl, partition=partition,
        validation_pct=a.val_pct, seed=a.seed,
        require_license=not a.allow_unknown_license,
        min_clinical_hits=a.min_clinical_hits,
        shuffle_buffer=a.shuffle_buffer if partition == "train" else 0,
        rank=rank, world_size=world, repeat=(a.repeat if partition == "train" else False))
    return DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                      pin_memory=device.type == "cuda")


@torch.no_grad()
def evaluate(model, data_loader, device, bf16, limit):
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    for i, data in enumerate(data_loader):
        if i >= limit:
            break
        ids, labels = data[:, :-1].to(device), data[:, 1:].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            ce = model(ids, labels)["ce_loss"]
        ntok = labels.numel()
        weighted_loss += ce.item() * ntok
        tokens += ntok
    acc = torch.tensor([weighted_loss, float(tokens)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(acc)
    model.train()
    ce = acc[0].item() / acc[1].item() if acc[1].item() else float("nan")
    return {"validation_ce": ce, "validation_perplexity": math.exp(min(40, ce)) if math.isfinite(ce) else None,
            "validation_tokens": int(acc[1].item())}


def run(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.steps is None and a.target_tokens is None:
        p.error("Specify --steps or --target-tokens")
    if min(a.seq_len, a.batch_size, a.grad_accum) < 1 or not 0 < a.val_pct < 100:
        p.error("seq-len/batch-size/grad-accum must be positive; val-pct 1..99")
    if a.source == "local" and not a.local_jsonl:
        p.error("--local-jsonl is required with --source local")
    cfg = ModelConfig.from_json(a.config)
    if a.seq_len > cfg.max_seq_len:
        p.error("--seq-len exceeds model config max_seq_len")
    tokenizer = get_tokenizer(a.tokenizer)
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        p.error("Tokenizer vocab size and model vocab_size MUST match")
    rank, world, device = init_distributed(a.strategy, a.device)
    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    if a.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("--bf16 requires a BF16-capable CUDA GPU")
    tokens_per_step = a.seq_len * a.batch_size * a.grad_accum * world
    steps = a.steps or math.ceil(a.target_tokens / tokens_per_step)
    if a.target_tokens:
        steps = min(steps, math.ceil(a.target_tokens / tokens_per_step))
    if a.strategy == "ddp":
        from mixtral_lab import MixtralLM
        model = DistributedDataParallel(MixtralLM(cfg).to(device), device_ids=[device.index],
                                        find_unused_parameters=True)
    else:
        model = make_model(cfg, a.strategy, device)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    first = resume_checkpoint(model, optim, a.resume, a.strategy, device)
    train_data = loader_for(a, "train", rank, world, device)
    train_it = iter(train_data)
    log = MetricsLog(a.metrics_log or Path(a.out) / "metrics.jsonl") if rank == 0 else None
    if rank == 0:
        core = model.module if isinstance(model, DistributedDataParallel) else model
        log({"stage": "healthcare-pretrain-from-scratch", "pretrained_model_used": False,
            "total_parameters": core.count_parameters(), "active_parameters_estimate": core.active_parameter_estimate(),
            "tokens_per_optimizer_step": tokens_per_step, "steps": steps,
            "healthcare_source": a.dataset if a.source == "hf" else a.local_jsonl,
            "update_all_parameters": True})
    start = time.monotonic()
    for step in range(first, steps):
        optim.zero_grad(set_to_none=True)
        mean_ce = 0.0
        for micro in range(a.grad_accum):
            try:
                packed = next(train_it).to(device, non_blocking=True)
            except StopIteration as e:
                raise RuntimeError("Corpus exhausted. Add more healthcare data or use --repeat") from e
            ids, y = packed[:, :-1], packed[:, 1:]
            nosync = (a.strategy == "ddp" and micro < a.grad_accum - 1)
            with (model.no_sync() if nosync else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    out = model(ids, y)
                    loss = out["loss"] / a.grad_accum
                loss.backward()
            mean_ce += out["ce_loss"].detach().item() / a.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        lr = lr_schedule(step, steps, a.warmup_steps, a.lr)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()
        if rank == 0 and (step == first or (step + 1) % 10 == 0):
            log({"stage": "pretrain", "step": step + 1, "train_ce": mean_ce, "lr": lr,
                "tokens_consumed": (step + 1) * tokens_per_step,
                "tokens_per_sec": int((step + 1 - first) * tokens_per_step / max(0.001, time.monotonic() - start))})
        if a.eval_every and (step + 1) % a.eval_every == 0:
            result = evaluate(model, loader_for(a, "validation", rank, world, device),
                              device, a.bf16, a.eval_batches)
            if rank == 0:
                log({"stage": "pretrain-validation", "step": step + 1, **result})
        if a.save_every and (step + 1) % a.save_every == 0:
            save_checkpoint(model, optim, step + 1, a, rank)
    save_checkpoint(model, optim, steps, a, rank)
    if rank == 0:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        manifest = {"stage": 1, "initialized_from_scratch": not bool(a.resume),
            "optimizer_steps": steps, "total_tokens_processed": steps * tokens_per_step,
            "dataset": a.dataset if a.source == "hf" else a.local_jsonl,
            "tokenizer": a.tokenizer, "config": a.config,
            "license_metadata_filter": not a.allow_unknown_license,
            "min_clinical_hits": a.min_clinical_hits,
            "notes": "Token count is processed tokens, not unique tokens. Exact dataset cursor is not checkpointed."}
        (Path(a.out) / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        print("Stage 1 saved:", a.out, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
