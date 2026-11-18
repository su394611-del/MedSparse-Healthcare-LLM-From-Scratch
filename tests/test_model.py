"""Small CPU tests validating causality, routing gradients, caching and positions."""
import pytest
import torch
from mixtral_lab import ModelConfig, MixtralLM
from mixtral_lab.model import rotary


def tiny(**changes):
    d = dict(vocab_size=260, hidden_size=64, num_layers=2, num_heads=4,
             num_kv_heads=2, intermediate_size=128, num_experts=4, top_k=2,
             max_seq_len=2_097_152, sliding_window=8, attention_chunk=4,
             attention_mode="sliding")
    d.update(changes)
    return ModelConfig(**d)


def test_forward_and_backward():
    torch.manual_seed(3)
    model = MixtralLM(tiny())
    ids = torch.randint(0, 260, (2, 15))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 15, 260)
    assert torch.isfinite(out["loss"])
    assert out["aux_loss"].item() > 0
    out["loss"].backward()
    assert model.layers[0].moe.gate.weight.grad is not None
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert any(e.w1.weight.grad is not None for e in model.layers[0].moe.experts)


def test_causal_no_future_leak():
    torch.manual_seed(4)
    m = MixtralLM(tiny()).eval()
    ids = torch.randint(0, 260, (1, 12))
    changed = ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 19) % 260
    with torch.no_grad():
        a = m(ids)["logits"][:, :8]
        b = m(changed)["logits"][:, :8]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_cache_matches_sliding_full_forward():
    torch.manual_seed(5)
    model = MixtralLM(tiny()).eval()
    ids = torch.randint(0, 260, (1, 17))
    with torch.no_grad():
        full = model(ids)["logits"]
        past = None
        pieces = []
        for s, e in ((0, 3), (3, 6), (6, 12), (12, 17)):
            out = model(ids[:, s:e], offset=s, past_key_values=past, use_cache=True)
            pieces.append(out["logits"])
            past = out["past_key_values"]
        from_cache = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, from_cache, rtol=1e-5, atol=2e-5)
    assert past[0][0].shape[-2] <= 7


def test_arbitrary_million_positions():
    torch.manual_seed(6)
    cfg = tiny()
    model = MixtralLM(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 260, (1, 3)), offset=2_097_145)
    assert bool(torch.isfinite(out["logits"]).all())
    z = rotary(torch.randn(1, 4, 2, 16), torch.tensor([1_000_000, 1_000_001]), 1e6)
    assert bool(torch.isfinite(z).all())
    with pytest.raises(ValueError):
        model(torch.randint(0, 260, (1, 3)), offset=2_097_151)


def test_full_attention_and_config_validation():
    model = MixtralLM(tiny(attention_mode="global"))
    output = model(torch.randint(0, 260, (1, 9)))
    assert output["logits"].shape == (1, 9, 260)
    with pytest.raises(ValueError):
        tiny(top_k=5)
