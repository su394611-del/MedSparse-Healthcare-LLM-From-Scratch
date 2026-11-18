"""Model architecture and validation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 14336
    num_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 32768
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    attention_mode: str = "global"  # global or sliding
    sliding_window: int = 4096
    attention_chunk: int = 256
    aux_loss_coef: float = 0.02
    tie_embeddings: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self):
        positive = ("vocab_size", "hidden_size", "num_layers", "num_heads", "num_kv_heads",
                    "intermediate_size", "num_experts", "top_k", "max_seq_len",
                    "sliding_window", "attention_chunk")
        for field in positive:
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.attention_mode not in {"global", "sliding"}:
            raise ValueError("attention_mode must be global or sliding")
        if self.rope_theta <= 1 or self.aux_loss_coef < 0:
            raise ValueError("invalid rope_theta or aux_loss_coef")

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
