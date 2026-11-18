"""Research-grade, trainable Mixtral-style sparse MoE foundation model."""
from .config import ModelConfig
from .model import MixtralLM

__all__ = ["ModelConfig", "MixtralLM"]
