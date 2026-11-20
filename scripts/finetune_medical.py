"""Full-parameter medical supervised fine-tuning for the native MixtralLM.

Requires a PRETRAINED, matching native checkpoint (weights only are imported).
Single/DDP: *.pt made by scripts.pretrain_healthcare; FSDP2: DCP checkpoint directory.
The --allow-random-init flag exists only for offline pipeline validation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mixtral_lab import MixtralLM, ModelConfig
from mixtral_lab.data import get_tokenizer
from mixtral_lab.medical_data import MedicalSFTDataset, DEFAULT_HF_DATASET, pad_medical_batch
from mixtral_lab.train_utils import (init_distributed, make_model, lr_schedule, save_checkpoint,
                                    resume_checkpoint, MetricsLog)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/moe_1b.json")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--pretrained", help="Native PyTorch .pt file OR FSDP2 DCP checkpoint directory")
    p.add_argument("--allow-random-init", action="store_true", help="PIPELINE TEST ONLY: not real fine-tuning")
    p.add_argument("--resume", help="Resume this medical fine-tuning run including optimizer")
    p.add_argument("--source", choices=["hf", "local"], default="hf")
    p.add_argument("--local-jsonl", help="Local JSON Lines with input/output fields")
    p.add_argument("--dataset", default=DEFAULT_HF_DATASET)
    p.add_argument("--dataset-config")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--schema", choices=["input_output", "instruction_input_output", "question_response"],
                   default="input_output")
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp2"], default="single")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Force CPU to train models larger than 4GB GPU VRAM")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--out", default="checkpoints/medical_sft")
    p.add_argument("--metrics-log", help="JSON Lines file for per-step metrics (default: <out>/metrics.jsonl)")
    a = p.parse_args(argv)
    if a.pretrained and a.resume:
        p.error("Choose --pretrained for first medical run OR --resume for continuing medical run")
    if not (a.pretrained or a.resume or a.allow_random_init):
        p.error("Real full fine-tuning requires --pretrained. For pipeline testing only, pass --allow-random-init")
    if a.source == "local" and not a.local_jsonl:
        p.error("--local-jsonl is mandatory with --source local")
    if min(a.steps, a.batch_size, a.grad_accum, a.seq_len, a.eval_batches) < 1:
        p.error("steps, batch-size, grad-accum, seq-len, eval-batches must be positive")
    if a.seq_len < 8:
        p.error("seq-len must be at least 8")
    if not 0 < a.val_pct < 100:
        p.error("val-pct must be between 1 and 99")
    return a


def load_pretrained(model, checkpoint, strategy):
    """Load weights ONLY. New AdamW optimizer is intentionally freshly initialized."""
    if strategy == "fsdp2":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        weights = get_model_state_dict(model)
        dcp.load({"model": weights}, checkpoint_id=str(checkpoint))
        set_model_state_dict(model, weights)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw.load_state_dict(weights, strict=True)


def make_data_loader(a, tokenizer, rank, world, partition, device):
    data = MedicalSFTDataset(tokenizer_path=a.tokenizer, seq_len=a.seq_len,
                             source=a.source, local_jsonl=a.local_jsonl,
                             dataset=a.dataset, dataset_config=a.dataset_config,
                             hf_split=a.hf_split, schema=a.schema,
                             partition=partition, val_pct=a.val_pct, seed=a.seed,
                             world_size=world, rank=rank,
                             repeat=partition == "train", shuffle_buffer=a.shuffle_buffer)
    return DataLoader(data, batch_size=a.batch_size, num_workers=a.num_workers,
                      collate_fn=lambda samples: pad_medical_batch(samples, tokenizer.token_to_id("<pad>")),
                      pin_memory=device.type == "cuda")


def evaluate(model, loader, device, max_batches, use_bf16):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            if step >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                ce = model(x, y)["ce_loss"]
            if torch.isfinite(ce):
                total_loss += float(ce) * int((y != -100).sum())
                count += int((y != -100).sum())
    stat = torch.tensor([total_loss, float(count)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stat, op=dist.ReduceOp.SUM)
    model.train()
    loss = float(stat[0] / stat[1]) if stat[1] > 0 else float("nan")
    return {"val_completion_ce": round(loss, 5),
            "val_completion_ppl": round(math.exp(min(40, loss)), 4) if math.isfinite(loss) else None,
            "val_supervised_tokens": int(stat[1].item())}


def run(argv=None):
    a = parse_args(argv)
    cfg = ModelConfig.from_json(a.config)
    if a.seq_len > cfg.max_seq_len:
        raise ValueError("--seq-len exceeds config max_seq_len")
    tok = get_tokenizer(a.tokenizer)
    if tok.get_vocab_size() != cfg.vocab_size:
        raise ValueError(f"Tokenizer vocab={tok.get_vocab_size()} but model config={cfg.vocab_size}; "
                         "MUST reuse original pretrained tokenizer")
    rank, world, device = init_distributed(a.strategy, a.device)
    # Shared initialization; pretrained values subsequently replace model weights.
    torch.manual_seed(a.seed)
    if a.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("--bf16 requires a CUDA GPU with BF16 support")
    # DDP may have unused experts in a microbatch; configure DDP accordingly.
    if a.strategy == "ddp":
        model = DistributedDataParallel(MixtralLM(cfg).to(device),
                                        device_ids=[device.index], find_unused_parameters=True)
    else:
        model = make_model(cfg, a.strategy, device)
    if a.pretrained:
        load_pretrained(model, a.pretrained, a.strategy)
    elif a.allow_random_init and rank == 0:
        print("WARNING: randomly initialized model; this only validates the training pipeline.", flush=True)
    assert all(p.requires_grad for p in model.parameters()), "Full fine-tuning requires all weights unfrozen"
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    first = resume_checkpoint(model, optimizer, a.resume, a.strategy, device) if a.resume else 0
    train_loader = make_data_loader(a, tok, rank, world, "train", device)
    val_loader = make_data_loader(a, tok, rank, world, "validation", device)
    train_it = iter(train_loader)
    model.train()
    log = MetricsLog(a.metrics_log or Path(a.out) / "metrics.jsonl") if rank == 0 else None
    if rank == 0:
        log({"stage": "medical-sft", "total_params": (model.module if isinstance(model, DistributedDataParallel) else model).count_parameters(),
                          "update_all_weights": True, "dataset": a.dataset if a.source == "hf" else a.local_jsonl,
                          "epochs_are_streaming": True, "world_size": world})
    try:
        for step in range(first, a.steps):
            optimizer.zero_grad(set_to_none=True)
            numerator, denom = 0.0, 0
            for micro in range(a.grad_accum):
                x, y = next(train_it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                no_sync = a.strategy == "ddp" and micro < a.grad_accum - 1
                with (model.no_sync() if no_sync else nullcontext()):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                        outputs = model(x, y)
                        loss = outputs["loss"] / a.grad_accum
                    loss.backward()
                numerator += float(outputs["ce_loss"].detach())
                denom += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            lr = lr_schedule(step, a.steps, a.warmup_steps, a.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if rank == 0 and (step == first or (step + 1) % 10 == 0):
                log({"stage": "sft", "step": step + 1, "train_completion_ce": round(numerator / denom, 6),
                     "lr": lr})
            if a.eval_every and (step + 1) % a.eval_every == 0:
                info = evaluate(model, val_loader, device, a.eval_batches, a.bf16)
                if rank == 0:
                    log({"stage": "sft-validation", "step": step + 1, **info})
            if a.save_every and (step + 1) % a.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, a, rank)
        if not a.save_every or (a.steps % a.save_every):
            save_checkpoint(model, optimizer, a.steps, a, rank)
        if rank == 0:
            target = Path(a.out)
            target.mkdir(parents=True, exist_ok=True)
            (target / "training_manifest.json").write_text(json.dumps({
                "stage": 2, "objective": "full-parameter-medical-sft",
                "pretrained_checkpoint_from_stage1": a.pretrained,
                "resumed_training": a.resume,
                "random_init_smoke_only": a.allow_random_init,
                "all_parameters_trainable": True,
                "optimizer": "new_adamw_on_first_sft_run",
                "config": a.config, "tokenizer": a.tokenizer,
                "dataset": a.dataset if a.source == "hf" else a.local_jsonl,
                "schema": a.schema, "steps": a.steps,
                "notes": "Medical quality and safety not implied by completion."
            }, indent=2), encoding="utf8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
