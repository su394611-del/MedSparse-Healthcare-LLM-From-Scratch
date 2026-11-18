"""Perplexity on a held-out local/Hugging Face split (native single-rank .pt)."""
from __future__ import annotations
import argparse
import math
import torch
from torch.utils.data import DataLoader
from mixtral_lab import MixtralLM, ModelConfig
from mixtral_lab.data import PackedTokens, get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--tokenizer", default="data/byte_tokenizer.json")
    p.add_argument("--checkpoint", default="checkpoints/smoke/last.pt")
    p.add_argument("--dataset", default="local")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--local-text", default="data/demo.txt")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=5)
    a = p.parse_args()
    cfg = ModelConfig.from_json(a.config)
    cfg.gradient_checkpointing = False
    tok = get_tokenizer(a.tokenizer)
    if tok.get_vocab_size() != cfg.vocab_size:
        p.error("Tokenizer and model vocab sizes differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixtralLM(cfg).to(device)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    data = PackedTokens(tokenizer_path=a.tokenizer, seq_len=a.seq_len,
                        dataset=a.dataset, dataset_config=a.dataset_config,
                        split=a.split, local_text=a.local_text,
                        shuffle_buffer=0, repeat=False)
    dataloader = DataLoader(data, batch_size=a.batch_size)
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, arr in enumerate(dataloader):
            if i >= a.batches:
                break
            x = arr[:, :-1].to(device)
            y = arr[:, 1:].to(device)
            out = model(x, labels=y)
            total_nll += float(out["ce_loss"]) * y.numel()
            total_tokens += y.numel()
    if not total_tokens:
        raise RuntimeError("No evaluation sequences available")
    avg = total_nll / total_tokens
    print(f"Evaluated tokens={total_tokens:,}; CE={avg:.4f}; perplexity={math.exp(avg):.4f}")
    print("Use a genuinely held-out dataset for meaningful results; do not evaluate on train data.")


if __name__ == "__main__":
    main()
