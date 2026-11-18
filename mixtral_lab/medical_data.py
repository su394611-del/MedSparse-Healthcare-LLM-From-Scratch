"""Supervised medical instruction examples with completion-only causal LM labels.

No downloaded medical records are included in this project. The default source is
Hugging Face Medical Intelligence 76K. All outputs require human validation before
clinical use. This module does NOT claim to anonymize or de-identify input data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from mixtral_lab.data import get_tokenizer

DEFAULT_HF_DATASET = "huzaifa525/Medical_Intelligence_Dataset_76k_2026_Edition"


def format_example(row: dict[str, Any], schema: str = "input_output") -> tuple[str, str]:
    """Supported schemas: input/output, instruction/input/output, question/response.

    Strings only. Reject rows with empty answers rather than training empty targets.
    """
    if schema == "input_output":
        question = row.get("input", "")
        answer = row.get("output", "")
    elif schema == "instruction_input_output":
        instruction = row.get("instruction", "")
        context = row.get("input", "")
        question = f"{instruction}\n\n{context}" if context else instruction
        answer = row.get("output", "")
    elif schema == "question_response":
        question = row.get("question", row.get("Question", ""))
        answer = row.get("response", row.get("Response", ""))
    else:
        raise ValueError(f"Unknown schema: {schema}")
    if not isinstance(question, str) or not isinstance(answer, str):
        return "", ""
    return question.strip(), answer.strip()


def bucket_for_question(question: str, *, seed: int = 42, val_pct: int = 5) -> str:
    """Stable split based on normalized question (avoids identical prompts crossing splits)."""
    if not 0 < val_pct < 100:
        raise ValueError("val_pct must be between 1 and 99")
    canonical = " ".join(question.casefold().split())
    h = hashlib.blake2b(f"{seed}:{canonical}".encode("utf-8"), digest_size=8).digest()
    return "validation" if int.from_bytes(h, "big") % 100 < val_pct else "train"


def local_or_hf_rows(*, source: str, path: str | None, dataset: str, dataset_config: str | None,
                     hf_split: str, shuffle_buffer: int, seed: int):
    if source == "local":
        if not path:
            raise ValueError("--local-jsonl is mandatory for local source")
        with Path(path).open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"local JSONL line {lineno} must contain a JSON object")
                yield row
        return
    if source != "hf":
        raise ValueError("source must be hf or local")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to stream medical data") from exc
    ds = load_dataset(dataset, name=dataset_config, split=hf_split, streaming=True)
    if shuffle_buffer > 1:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from ds


def tokenize_example(tokenizer, prompt: str, answer: str, *, seq_len: int,
                     system_prompt: str = "Provide accurate educational medical information, and acknowledge uncertainty."):
    """Make next-token targets while masking user/system portions of the prompt.

    For a long sample: keep the last prompt tokens + a reasonably sized answer.
    Truncate answer only if it alone exceeds the budget. Training seq_len is the
    input length; token count including the last target is seq_len + 1.

    Only EOS is required for sample tokenization: padding is handled separately
    by pad_medical_batch(), which can safely use token ID 0 when <pad> is absent.
    This avoids changing vocabulary size after pretraining.
    """
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("Tokenizer must include an <eos> special token")
    header = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
    prompt_ids = ([bos] if bos is not None else []) + tokenizer.encode(header).ids
    answer_ids = tokenizer.encode(answer).ids + [eos]
    budget = seq_len + 1
    if len(answer_ids) > budget - 2:
        answer_ids = answer_ids[: budget - 2]
        answer_ids[-1] = eos
    remaining = budget - len(answer_ids)
    prompt_ids = prompt_ids[-remaining:]
    if not prompt_ids or not answer_ids:
        raise ValueError("Insufficient room for prompt and response")
    tokens = prompt_ids + answer_ids
    targets = [-100] * len(prompt_ids) + answer_ids
    x = torch.tensor(tokens[:-1], dtype=torch.long)
    y = torch.tensor(targets[1:], dtype=torch.long)
    assert (y != -100).any().item(), "Every sample must have at least one answer token"
    return x, y


class MedicalSFTDataset(IterableDataset):
    def __init__(self, *, tokenizer_path: str, seq_len: int, source: str = "hf",
                 local_jsonl: str | None = None, dataset: str = DEFAULT_HF_DATASET,
                 dataset_config: str | None = None, hf_split: str = "train",
                 schema: str = "input_output", partition: str = "train",
                 val_pct: int = 5, seed: int = 42, world_size: int = 1,
                 rank: int = 0, repeat: bool = False, shuffle_buffer: int = 1000,
                 max_examples: int | None = None, system_prompt: str | None = None):
        super().__init__()
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation")
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.source = source
        self.local_jsonl = local_jsonl
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.hf_split = hf_split
        self.schema = schema
        self.partition = partition
        self.val_pct = val_pct
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.repeat = repeat
        self.shuffle_buffer = shuffle_buffer
        self.max_examples = max_examples
        self.system_prompt = system_prompt

    def __iter__(self):
        wi = get_worker_info()
        workers = wi.num_workers if wi else 1
        wid = wi.id if wi else 0
        rank_count = self.world_size * workers
        shard_index = self.rank * workers + wid
        tokenizer = get_tokenizer(self.tokenizer_path)
        if tokenizer.token_to_id("<eos>") is None:
            raise ValueError("Tokenizer missing <eos>; cannot train completion-only SFT")
        epoch = 0
        while True:
            rows = local_or_hf_rows(source=self.source, path=self.local_jsonl,
                                   dataset=self.dataset, dataset_config=self.dataset_config,
                                   hf_split=self.hf_split, shuffle_buffer=self.shuffle_buffer
                                   if self.partition == "train" else 0,
                                   seed=self.seed + epoch)
            seen = 0
            emitted = 0
            rejected = 0
            first_error: Exception | None = None
            for row in rows:
                q, a = format_example(row, self.schema)
                if not q or not a:
                    continue
                if bucket_for_question(q, seed=self.seed, val_pct=self.val_pct) != self.partition:
                    continue
                seen += 1
                if (seen - 1) % rank_count != shard_index:
                    continue
                try:
                    example = tokenize_example(
                        tokenizer, q, a, seq_len=self.seq_len,
                        **({"system_prompt": self.system_prompt} if self.system_prompt else {})
                    )
                except Exception as exc:
                    # Some older BPE tokenizers declare unk_token="<unk>" without
                    # including <unk> in their learned vocabulary. Encoding a
                    # previously unseen character then raises a generic Exception,
                    # not ValueError. Skip ONLY this known data/tokenizer error;
                    # do not hide unrelated failures such as CUDA or I/O errors.
                    missing_unk = (
                        "Unk token" in str(exc)
                        and "not found in the vocabulary" in str(exc)
                    )
                    if not isinstance(exc, ValueError) and not missing_unk:
                        raise
                    rejected += 1
                    if first_error is None:
                        first_error = exc
                    if missing_unk and rejected <= 3:
                        print(
                            "[medical_data] Skipping example with unseen text: "
                            "tokenizer declares <unk> but has no usable <unk> ID. "
                            "No tokenizer IDs or model weights were changed.",
                            flush=True,
                        )
                    continue
                emitted += 1
                yield example
                if self.max_examples is not None and emitted >= self.max_examples:
                    return
            if not self.repeat:
                if emitted == 0 and first_error is not None:
                    raise RuntimeError(
                        f"No usable examples. {rejected} samples failed tokenization; "
                        f"first failure: {first_error}"
                    ) from first_error
                return
            if not seen:
                raise RuntimeError(
                    "Partition empty: no rows passed schema and split filtering. "
                    "Check --schema, --hf-split, dataset content and val_pct."
                )
            if emitted == 0:
                if first_error is not None:
                    raise RuntimeError(
                        f"No usable rows on worker {shard_index}/{rank_count}; "
                        f"{rejected} tokenization errors. First failure: {first_error}"
                    ) from first_error
                raise RuntimeError(
                    f"No rows assigned to worker {shard_index}/{rank_count}; "
                    "reduce world size/workers or provide more examples"
                )
            epoch += 1


def pad_medical_batch(samples, pad_id: int | None):
    """Right-pad inputs; padding labels are ignored by CrossEntropyLoss.

    If the tokenizer has no <pad> token, use existing ID 0 as a filler for
    right-padding. It is a valid vocabulary ID, so this does not resize the
    pretrained model; the corresponding target labels are always -100.
    """
    if not samples:
        raise ValueError("Batch must not be empty")
    if pad_id is None:
        pad_id = 0
    longest = max(x.numel() for x, _ in samples)
    x = torch.full((len(samples), longest), int(pad_id), dtype=torch.long)
    y = torch.full((len(samples), longest), -100, dtype=torch.long)
    for i, (xi, yi) in enumerate(samples):
        x[i, :len(xi)] = xi
        y[i, :len(yi)] = yi
    return x, y
