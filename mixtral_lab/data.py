"""Hugging Face streaming dataset -> packed, rank/worker-sharded token sequences."""
from __future__ import annotations
import itertools
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info


class ByteTokenizer:
    """Dependency-free, 260-token byte tokenizer for offline smoke tests only."""
    def token_to_id(self, token):
        return {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}.get(token)

    def get_vocab_size(self):
        return 260

    def encode(self, text):
        from types import SimpleNamespace
        return SimpleNamespace(ids=[4 + b for b in text.encode("utf-8")])

    def decode(self, ids, skip_special_tokens=True):
        data = bytes([i - 4 for i in ids if 4 <= i < 260])
        return data.decode("utf-8", errors="replace")


def get_tokenizer(path):
    # Small offline smoke test needs no Hugging Face downloads or dependencies.
    import json
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if content.get("type") == "built_in_byte_tokenizer":
        return ByteTokenizer()
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
    tok = Tokenizer.from_file(str(path))
    if tok.token_to_id("<eos>") is None:
        raise ValueError("Expected <eos> special token in tokenizer")
    return tok


def source_docs(dataset, dataset_config, split, local_text, *, shard_count=1,
                shard_index=0, shuffle_buffer=0, seed=42):
    if dataset == "local":
        if not local_text:
            raise ValueError("--local-text is required for --dataset local")
        path = Path(local_text)
        # For smoke tests, each nonblank line is a document.
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i % shard_count == shard_index and line.strip():
                    yield line.strip()
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install 'datasets' to stream Hugging Face data") from exc
        ds = load_dataset(dataset, name=dataset_config or None, split=split, streaming=True)
        if shuffle_buffer > 1:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        if shard_count > 1:
            ds = ds.shard(num_shards=shard_count, index=shard_index)
        for row in ds:
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield text


class PackedTokens(IterableDataset):
    def __init__(self, *, tokenizer_path, seq_len, dataset, dataset_config=None,
                 split="train", local_text=None, rank=0, world_size=1,
                 seed=42, repeat=False, shuffle_buffer=1000, max_doc_chars=100000):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.split = split
        self.local_text = local_text
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.repeat = repeat
        self.shuffle_buffer = shuffle_buffer
        self.max_doc_chars = max_doc_chars

    def __iter__(self):
        wi = get_worker_info()
        workers = wi.num_workers if wi else 1
        wid = wi.id if wi else 0
        total_shards = self.world_size * workers
        shard_index = self.rank * workers + wid
        tok = get_tokenizer(self.tokenizer_path)
        eos = tok.token_to_id("<eos>")
        packed = []
        epoch = 0
        while True:
            docs = source_docs(
                self.dataset, self.dataset_config, self.split, self.local_text,
                shard_count=total_shards, shard_index=shard_index,
                shuffle_buffer=self.shuffle_buffer if self.dataset != "local" else 0,
                seed=self.seed + epoch)
            any_seen = False
            for doc in docs:
                any_seen = True
                packed.extend(tok.encode(doc[:self.max_doc_chars]).ids)
                packed.append(eos)
                while len(packed) >= self.seq_len + 1:
                    arr = packed[:self.seq_len + 1]
                    packed = packed[self.seq_len:]
                    yield torch.tensor(arr, dtype=torch.long)
            if not self.repeat:
                return
            if not any_seen:
                raise RuntimeError("No text in this shard; reduce workers/ranks or use more data")
            epoch += 1


def tokenizer_iterator(dataset, config, split, local_text, sample_docs):
    return itertools.islice(source_docs(dataset, config, split, local_text), sample_docs)
