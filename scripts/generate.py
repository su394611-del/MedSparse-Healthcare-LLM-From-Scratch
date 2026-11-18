"""Autoregressive generation with chunked prefill and sliding-window KV cache."""
import argparse
import torch
from mixtral_lab import MixtralLM, ModelConfig
from mixtral_lab.data import get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--checkpoint", default="checkpoints/run/last.pt")
    p.add_argument("--prompt", default="Mixture of experts")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prefill-chunk", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    a = p.parse_args()
    if a.temperature <= 0 or a.top_k <= 0 or a.prefill_chunk <= 0:
        p.error("temperature, top-k and prefill-chunk must be positive")
    cfg = ModelConfig.from_json(a.config)
    if cfg.gradient_checkpointing:
        cfg.gradient_checkpointing = False
    tok = get_tokenizer(a.tokenizer)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixtralLM(cfg).to(dev)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = tok.encode(a.prompt).ids
    if not ids:
        ids = [tok.token_to_id("<bos>")]
    if len(ids) + a.max_new_tokens > cfg.max_seq_len:
        p.error("prompt plus generated tokens exceeds config max_seq_len")
    cache = None
    offset = 0
    with torch.inference_mode():
        for i in range(0, len(ids), a.prefill_chunk):
            piece = ids[i:i + a.prefill_chunk]
            out = model(torch.tensor([piece], device=dev), offset=offset,
                        past_key_values=cache, use_cache=True)
            cache = out["past_key_values"]
            offset += len(piece)
        for i in range(a.max_new_tokens):
            logits = out["logits"][0, -1].float() / a.temperature
            k = min(a.top_k, logits.numel())
            vals, choices = torch.topk(logits, k)
            pick = choices[torch.multinomial(torch.softmax(vals, -1), 1)].item()
            ids.append(pick)
            if pick == tok.token_to_id("<eos>"):
                break
            if i < a.max_new_tokens - 1:
                out = model(torch.tensor([[pick]], device=dev), offset=offset,
                            past_key_values=cache, use_cache=True)
                cache = out["past_key_values"]
                offset += 1
    print(tok.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
