"""Mixtral-inspired decoder LM; dense GQA attention + token-level top-k sparse experts.

This is NOT a reproduction of Mistral's pretrained weights or an optimized
expert-parallel implementation. An expert's weights reside on each process
unless additional (not included) tensor/pipeline/expert parallelism is used.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rotary(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """x shape [B,H,T,D]; RoPE works at arbitrary explicit nonnegative offsets."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    phase = torch.outer(positions.to(torch.float32), inv)
    c = torch.cos(phase).to(x.dtype)[None, None, :, :]
    s = torch.sin(phase).to(x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, *, offset: int = 0,
                past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        b, t, _ = x.shape
        h, hk, d = self.cfg.num_heads, self.cfg.num_kv_heads, self.head_dim
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, hk, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, hk, d).transpose(1, 2)
        positions = torch.arange(offset, offset + t, device=x.device)
        q, k = rotary(q, positions, self.cfg.rope_theta), rotary(k, positions, self.cfg.rope_theta)
        old_len = 0
        if past is not None:
            old_len = past[0].shape[-2]
            k = torch.cat([past[0], k], dim=-2)
            v = torch.cat([past[1], v], dim=-2)
        # GQA repeat: simple reference implementation; production kernels should
        # use native GQA or expert-aware fused kernels to avoid extra memory.
        repeat = h // hk
        kr = k.repeat_interleave(repeat, dim=1)
        vr = v.repeat_interleave(repeat, dim=1)

        if past is None and (self.cfg.attention_mode == "global" or t <= self.cfg.sliding_window):
            out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, dropout_p=0.0)
        elif past is not None:
            # Absolute positions avoid incorrectly masking cached KV when q_len != k_len.
            qpos = torch.arange(offset, offset + t, device=x.device)
            kpos = torch.arange(offset - old_len, offset + t, device=x.device)
            mask = (kpos[None, :] <= qpos[:, None])
            if self.cfg.attention_mode == "sliding":
                mask &= ((qpos[:, None] - kpos[None, :]) < self.cfg.sliding_window)
            out = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask, dropout_p=0.0)
        else:
            # Bounded local causal attention, O(T * (window + chunk)) score work,
            # NOT O(T^2), but stores training activations for all T positions.
            pieces = []
            w = self.cfg.sliding_window
            for qs in range(0, t, self.cfg.attention_chunk):
                qe = min(qs + self.cfg.attention_chunk, t)
                ks = max(0, qs - w + 1)
                qpos = torch.arange(qs, qe, device=x.device)
                kpos = torch.arange(ks, qe, device=x.device)
                mask = ((kpos[None, :] <= qpos[:, None]) &
                        (qpos[:, None] - kpos[None, :] < w))
                chunk = F.scaled_dot_product_attention(
                    q[:, :, qs:qe, :], kr[:, :, ks:qe, :], vr[:, :, ks:qe, :],
                    attn_mask=mask, dropout_p=0.0)
                pieces.append(chunk)
            out = torch.cat(pieces, dim=2)
        out = self.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1))
        cache = None
        if use_cache:
            if self.cfg.attention_mode == "sliding":
                keep = max(1, self.cfg.sliding_window - 1)
                cache = (k[:, :, -keep:, :].detach(), v[:, :, -keep:, :].detach())
            else:
                cache = (k.detach(), v.detach())  # Global KV grows linearly with context.
        return out, cache


class SwiGLUExpert(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        scores = self.gate(flat).float()
        probs = F.softmax(scores, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(flat.dtype)
        # Differentiable expert token dispatch; inefficient for huge expert parallel runs.
        output = torch.zeros_like(flat)
        for eid, expert in enumerate(self.experts):
            token_idx, choice_idx = torch.where(indices == eid)
            if token_idx.numel():
                update = expert(flat.index_select(0, token_idx))
                update = update * weights[token_idx, choice_idx].unsqueeze(-1)
                output.index_add_(0, token_idx, update)
        # Switch-style auxiliary load balance regularizer with mean top-k routing.
        with torch.no_grad():
            fraction = F.one_hot(indices, num_classes=self.num_experts).float().mean((0, 1))
        aux = self.num_experts * (probs.mean(0) * fraction).sum()
        return output.view(b, t, d), aux


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.moe = SparseMoE(cfg)

    def forward(self, x, offset=0, past=None, use_cache=False):
        att, cache = self.attn(self.attn_norm(x), offset=offset, past=past, use_cache=use_cache)
        x = x + att
        ffn, aux = self.moe(self.ffn_norm(x))
        return x + ffn, aux, cache


class MixtralLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)

    def reset_all_parameters(self):
        """Use after FSDP2 meta initialization and to_empty."""
        self.apply(self._initialize)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                *, offset: int = 0,
                past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, time]")
        b, t = input_ids.shape
        if t < 1 or offset < 0 or offset + t > self.cfg.max_seq_len:
            raise ValueError("input length/position exceeds configured max_seq_len")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("cache must contain one entry per layer")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must be aligned with input_ids")
        x = self.embed(input_ids)
        total_aux = x.new_zeros((), dtype=torch.float32)
        new_caches = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            if self.cfg.gradient_checkpointing and self.training and not use_cache:
                def wrapped(z, layer=layer):
                    y, aux, _ = layer(z, offset=offset)
                    return y, aux
                x, aux = checkpoint(wrapped, x, use_reentrant=False)
                cache = None
            else:
                x, aux, cache = layer(x, offset=offset, past=past, use_cache=use_cache)
            total_aux = total_aux + aux
            if use_cache:
                new_caches.append(cache)
        logits = self.lm_head(self.norm(x))
        aux = total_aux / len(self.layers)
        ce = None
        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                 labels.reshape(-1), ignore_index=-100)
            loss = ce + self.cfg.aux_loss_coef * aux
        return {"logits": logits, "loss": loss, "ce_loss": ce, "aux_loss": aux,
                "past_key_values": new_caches if use_cache else None}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def active_parameter_estimate(self):
        # Structural approximation: sum all non-expert weights + top-k experts/layer.
        non_expert = sum(p.numel() for name, p in self.named_parameters() if ".experts." not in name)
        expert_size = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters())
        return non_expert + self.cfg.num_layers * self.cfg.top_k * expert_size
