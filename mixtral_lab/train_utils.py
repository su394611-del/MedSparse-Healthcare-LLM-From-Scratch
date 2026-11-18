"""Distributed training, optimizer and checkpoint utilities shared by both HEALTHCARE stages."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mixtral_lab import MixtralLM, ModelConfig

def init_distributed(strategy, device_preference="auto"):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_preference must be auto, cpu, or cuda")
    if strategy != "single" and device_preference == "cpu":
        raise ValueError("This project supports CPU only for --strategy single")
    if strategy == "single" and world != 1:
        raise ValueError("Use --strategy ddp or --strategy fsdp2 under torchrun")
    if strategy != "single" and world < 2:
        raise ValueError("Run multi-GPU strategy with torchrun --nproc_per_node=N (N>=2)")
    if strategy != "single" and not torch.cuda.is_available():
        raise RuntimeError("Distributed strategies in this project require CUDA GPUs")
    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA GPU detected")
    use_cuda = device_preference != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, device


def make_model(cfg, strategy, device):
    if strategy == "fsdp2":
        # Meta init prevents allocating all 47B full-precision weights on every rank.
        from torch.distributed.fsdp import fully_shard
        with torch.device("meta"):
            model = MixtralLM(cfg)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        model.to_empty(device=device)
        model.reset_all_parameters()
    else:
        model = MixtralLM(cfg).to(device)
        if strategy == "ddp":
            model = DistributedDataParallel(model, device_ids=[device.index])
    return model


def lr_schedule(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def save_checkpoint(model, optimizer, step, args, rank):
    out = Path(args.out)
    if args.strategy == "fsdp2":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        ckpt = out / f"step_{step:08d}"
        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(ckpt))
        if rank == 0:
            (ckpt / "trainer.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    else:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            state = {"step": step, "model": raw_model.state_dict(),
                     "optimizer": optimizer.state_dict(), "config": raw_model.cfg.to_dict()}
            tmp = out / "last.tmp"
            torch.save(state, tmp)
            tmp.replace(out / "last.pt")
    if dist.is_initialized():
        dist.barrier()


def resume_checkpoint(model, optimizer, path, strategy, device):
    if not path:
        return 0
    if strategy == "fsdp2":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        state, optim = get_state_dict(model, optimizer)
        dcp.load({"model": state, "optimizer": optim}, checkpoint_id=str(path))
        set_state_dict(model, optimizer, model_state_dict=state, optim_state_dict=optim)
        return json.loads((Path(path) / "trainer.json").read_text())["step"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


