"""Streaming token packer can work completely offline."""
from pathlib import Path
from mixtral_lab.data import PackedTokens, get_tokenizer


def test_offline_byte_tokenizer_and_packing():
    root = Path(__file__).resolve().parents[1]
    tok = get_tokenizer(root / "data/byte_tokenizer.json")
    assert tok.decode(tok.encode("Mixtral and 🧠").ids) == "Mixtral and 🧠"
    data = PackedTokens(tokenizer_path=root / "data/byte_tokenizer.json",
                        dataset="local", local_text=root / "data/demo.txt",
                        seq_len=16, shuffle_buffer=0)
    first = next(iter(data))
    assert first.shape == (17,)
    assert first.min() >= 0 and first.max() < 260
