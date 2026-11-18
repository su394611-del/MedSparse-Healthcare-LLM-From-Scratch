"""Streaming, auditable healthcare-only next-token pretraining input.

Source restriction is intentionally strict: no FineWeb or general-web fallback.
Clinical keyword screening is a reproducible heuristic, NOT a clinical classifier.
The dataset must still be independently audited for copyright and patient privacy.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from mixtral_lab.data import get_tokenizer

HF_PRETRAIN_CORPORA = {
    "common-pile/pubmed_filtered",  # full text; metadata.license for each record
    "slinusc/PubMedAbstractsSubset",  # title + abstract; dataset card CC0; verify source rights
}
DEFAULT_HF_CORPUS = "common-pile/pubmed_filtered"
CLINICAL_TERMS = (
    "patient", "patients", "clinical", "medical", "healthcare", "hospital", "physician",
    "medicine", "disease", "diagnos", "treatment", "therap", "symptom", "syndrome",
    "infection", "infectious", "cancer", "tumor", "tumour", "oncolog", "diabet",
    "cardiac", "cardiolog", "heart", "stroke", "neurolog", "brain", "surgery",
    "blood", "respiratory", "pulmonary", "pharmacolog", "drug", "vaccine",
    "immunolog", "patholog", "epidemiolog", "pregnan", "pediatr", "renal",
    "kidney", "liver", "fracture", "fever", "antibiotic", "mortality",
)


def license_ok(value: Any) -> bool:
    """Allow a narrowly defined subset of CC0/public-domain/attribution licenses.

    This checks only metadata. Actual reuse permissions must still be verified.
    """
    if not isinstance(value, str):
        return False
    s = value.strip().lower().replace("_", "-")
    if any(t in s for t in ("noncommercial", "non-commercial", "by-nc", "sharealike", "share-alike", "by-sa", "all rights reserved", "by-nd")):
        return False
    if s in {"cc0", "cc0-1.0", "public domain", "cc-by-4.0", "cc-by-3.0", "cc-by-2.0"}:
        return True
    if "creativecommons.org/publicdomain/zero/1.0" in s:
        return True
    return bool(re.search(r"creativecommons\.org/licenses/by/(?:2\.0|3\.0|4\.0)(?:/|$)", s))


def extract_medical_text(row: dict, corpus: str) -> str:
    if corpus == "common-pile/pubmed_filtered":
        return row.get("text", "") if isinstance(row.get("text"), str) else ""
    if corpus == "slinusc/PubMedAbstractsSubset":
        title, abstract = row.get("title", ""), row.get("abstract", "")
        return "\n\n".join(v.strip() for v in (title, abstract) if isinstance(v, str) and v.strip())
    # For local JSONL: accept text or title + abstract.
    value = row.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "\n\n".join(v.strip() for v in (row.get("title"), row.get("abstract")) if isinstance(v, str) and v.strip())


def medical_filter(text: str, *, min_hits: int = 1) -> bool:
    s = text.lower()
    return sum(s.count(word) for word in CLINICAL_TERMS) >= min_hits


def row_id(row: dict, text: str) -> str:
    """Hash document text to keep identical records in the same split across sources."""
    canonical = " ".join(text.casefold().split())
    return hashlib.blake2b(canonical.encode("utf8"), digest_size=16).hexdigest()


def stable_partition(identifier: str, seed: int = 42, validation_pct: int = 2) -> str:
    if not 0 <= validation_pct < 100:
        raise ValueError("validation_pct must be 0..99")
    b = hashlib.blake2b(f"{seed}:{identifier}".encode(), digest_size=8).digest()
    return "validation" if int.from_bytes(b, "big") % 100 < validation_pct else "train"


def source_rows(source: str, dataset: str, dataset_config: str | None,
                split: str, local_jsonl: str | None, shuffle_buffer: int, seed: int) -> Iterator[dict]:
    if source == "local":
        if not local_jsonl:
            raise ValueError("--local-jsonl required when --source local")
        with Path(local_jsonl).open(encoding="utf8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Local JSONL line {n} is not an object")
                    yield row
    elif source == "hf":
        if dataset not in HF_PRETRAIN_CORPORA:
            raise ValueError(f"Not an approved healthcare pretraining source: {dataset}")
        from datasets import load_dataset
        ds = load_dataset(dataset, name=dataset_config or None, split=split, streaming=True)
        if shuffle_buffer > 1:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        yield from ds
    else:
        raise ValueError("source must be 'hf' or 'local'")


def healthcare_docs(*, source: str = "hf", dataset: str = DEFAULT_HF_CORPUS,
                    dataset_config: str | None = None, split: str = "train",
                    local_jsonl: str | None = None, partition: str = "train",
                    validation_pct: int = 2, seed: int = 42,
                    shuffle_buffer: int = 0, require_license: bool = True,
                    min_clinical_hits: int = 1, max_doc_chars: int = 100_000,
                    shard_count: int = 1, shard_index: int = 0) -> Iterator[str]:
    if partition not in {"train", "validation"}:
        raise ValueError("partition must be train or validation")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    for row in source_rows(source, dataset, dataset_config, split, local_jsonl, shuffle_buffer, seed):
        text = extract_medical_text(row, dataset if source == "hf" else "local")
        if not isinstance(text, str) or not text.strip():
            continue
        if min_clinical_hits > 0 and not medical_filter(text, min_hits=min_clinical_hits):
            continue
        if require_license:
            metadata = row.get("metadata") or {}
            lic = metadata.get("license") if isinstance(metadata, dict) else None
            lic = lic or row.get("license")
            if source == "hf" and dataset == "slinusc/PubMedAbstractsSubset":
                # Declared dataset-level CC0. Underlying articles' rights may differ.
                lic = lic or "cc0-1.0"
            if not license_ok(lic):
                continue
        doc_hash = row_id(row, text)
        if stable_partition(doc_hash, seed, validation_pct) != partition:
            continue
        # Assignment by hash is deterministic across worker counts and iteration order.
        if int(doc_hash[:16], 16) % shard_count != shard_index:
            continue
        yield text[:max_doc_chars]


class HealthcarePackedTokens(IterableDataset):
    """Pack streamed healthcare documents into seq_len+1 token blocks."""
    def __init__(self, *, tokenizer_path: str, seq_len: int, source: str = "hf",
                 dataset: str = DEFAULT_HF_CORPUS, dataset_config: str | None = None,
                 split: str = "train", local_jsonl: str | None = None,
                 partition: str = "train", validation_pct: int = 2,
                 seed: int = 42, shuffle_buffer: int = 1000,
                 require_license: bool = True, min_clinical_hits: int = 1,
                 max_doc_chars: int = 100_000, rank: int = 0, world_size: int = 1,
                 repeat: bool = False):
        super().__init__()
        self.kwargs = dict(source=source, dataset=dataset, dataset_config=dataset_config,
                           split=split, local_jsonl=local_jsonl, partition=partition,
                           validation_pct=validation_pct, seed=seed,
                           shuffle_buffer=shuffle_buffer, require_license=require_license,
                           min_clinical_hits=min_clinical_hits, max_doc_chars=max_doc_chars)
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat

    def __iter__(self):
        wi = get_worker_info()
        nworkers, worker_id = (wi.num_workers, wi.id) if wi else (1, 0)
        tokenizer = get_tokenizer(self.tokenizer_path)
        eos = tokenizer.token_to_id("<eos>")
        if eos is None:
            raise ValueError("Tokenizer must contain <eos>")
        epoch = 0
        while True:
            kwargs = self.kwargs.copy()
            kwargs["shard_count"] = nworkers * self.world_size
            kwargs["shard_index"] = self.rank * nworkers + worker_id
            buf = []
            count = 0
            for doc in healthcare_docs(**kwargs):
                count += 1
                buf.extend(tokenizer.encode(doc).ids)
                buf.append(eos)
                while len(buf) >= self.seq_len + 1:
                    yield torch.tensor(buf[:self.seq_len + 1], dtype=torch.long)
                    buf = buf[self.seq_len:]
            if not self.repeat:
                return
            if count == 0:
                raise RuntimeError("No eligible clinical documents in shard. Check filters/splits/license policy.")
            epoch += 1
