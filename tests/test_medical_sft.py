"""Offline tests for medical SFT using tiny random weights and JSONL fixtures."""
from __future__ import annotations
import json
import torch

from mixtral_lab import MixtralLM, ModelConfig
from mixtral_lab.data import get_tokenizer
from mixtral_lab.medical_data import (bucket_for_question, format_example,
                                      tokenize_example, pad_medical_batch, MedicalSFTDataset)


def test_medical_schema_variants():
    assert format_example({"input": "Q?", "output": "A"}) == ("Q?", "A")
    assert format_example({"instruction": "Explain", "input": "X", "output": "Y"},
                          "instruction_input_output") == ("Explain\n\nX", "Y")
    assert format_example({"question": "Q", "response": "A"}, "question_response") == ("Q", "A")
    assert format_example({"input": "", "output": "A"}) == ("", "A")


def test_completion_mask_and_eos():
    tok = get_tokenizer("data/byte_tokenizer.json")
    prompt = "What is blood pressure?"
    answer = "Pressure in arteries."
    x, y = tokenize_example(tok, prompt, answer, seq_len=256)
    header = ("### System:\nProvide accurate educational medical information, and acknowledge uncertainty."
              "\n\n### User:\n" + prompt + "\n\n### Assistant:\n")
    prompt_length = len(tok.encode(header).ids) + 1
    assert (y[:prompt_length-1] == -100).all()
    assert y[prompt_length-1] == tok.encode(answer).ids[0]
    assert y[-1].item() == tok.token_to_id("<eos>")
    assert (y != -100).sum().item() == len(tok.encode(answer).ids) + 1
    xx, yy = pad_medical_batch([(x, y), (x[:-5], y[:-5])], tok.token_to_id("<pad>"))
    assert xx.shape[0] == 2 and yy.shape == xx.shape
    assert (yy[1, -5:] == -100).all()


def test_sft_backprop_updates_all_selected_expert_types():
    cfg = ModelConfig(vocab_size=260, hidden_size=64, num_layers=1, num_heads=4,
                      num_kv_heads=2, intermediate_size=128, num_experts=4,
                      top_k=2, max_seq_len=256, attention_mode="global")
    torch.manual_seed(1)
    model = MixtralLM(cfg)
    tok = get_tokenizer("data/byte_tokenizer.json")
    x, y = tokenize_example(tok, "What causes influenza?", "Influenza viruses.", seq_len=128)
    old = model.embed.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(x[None], y[None])["loss"]
    loss.backward()
    assert model.embed.weight.grad is not None
    assert model.layers[0].moe.gate.weight.grad is not None
    assert any(expert.w2.weight.grad is not None for expert in model.layers[0].moe.experts)
    opt.step()
    assert not torch.equal(old, model.embed.weight.detach())


def test_stable_split_and_local_stream(tmp_path):
    file = tmp_path / "examples.jsonl"
    file.write_text("\n".join(json.dumps({"input": f"Medical question {i}", "output": f"Answer {i}"})
                              for i in range(100)) + "\n", encoding="utf-8")
    assert bucket_for_question(" Question   ONE ", seed=3) == bucket_for_question("question one", seed=3)
    a = MedicalSFTDataset(tokenizer_path="data/byte_tokenizer.json", seq_len=128,
                          source="local", local_jsonl=str(file), partition="train", val_pct=25)
    b = MedicalSFTDataset(tokenizer_path="data/byte_tokenizer.json", seq_len=128,
                          source="local", local_jsonl=str(file), partition="validation", val_pct=25)
    assert len(list(a)) + len(list(b)) == 100
