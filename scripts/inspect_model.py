"""Count billions of parameters without allocating full tensors or loading weights."""
import argparse
import torch
from mixtral_lab import MixtralLM, ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mixtral_8x7b_reference.json")
    p.add_argument("--check-rope-million", action="store_true")
    a = p.parse_args()
    cfg = ModelConfig.from_json(a.config)
    with torch.device("meta"):
        model = MixtralLM(cfg)
    print("Configuration:", a.config)
    print(f"Total parameters: {model.count_parameters():,} ({model.count_parameters()/1e9:.3f}B)")
    print(f"Estimated active/token: {model.active_parameter_estimate():,} ({model.active_parameter_estimate()/1e9:.3f}B)")
    print(f"Max position setting: {cfg.max_seq_len:,}")
    print(f"Attention: {cfg.attention_mode}")
    if a.check_rope_million:
        from mixtral_lab.model import rotary
        z = torch.randn(1, 2, 16, cfg.hidden_size // cfg.num_heads)
        pos = torch.arange(1_048_560, 1_048_576)
        y = rotary(z, pos, cfg.rope_theta)
        print(f"RoPE numerical position check: {bool(torch.isfinite(y).all())} at positions 1,048,560..1,048,575")
        print("This is NOT a million-token forward pass or proof of long-context quality.")


if __name__ == "__main__":
    main()
