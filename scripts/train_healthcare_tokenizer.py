"""Healthcare-only BPE tokenizer. Never reads general-web or FineWeb datasets."""
from __future__ import annotations
import argparse
from itertools import islice
from pathlib import Path

from mixtral_lab.healthcare_corpus import healthcare_docs, DEFAULT_HF_CORPUS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["hf", "local"], default="hf")
    p.add_argument("--dataset", default=DEFAULT_HF_CORPUS)
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="train")
    p.add_argument("--local-jsonl")
    p.add_argument("--sample-docs", type=int, default=200_000)
    p.add_argument("--vocab-size", type=int, default=32_000)
    p.add_argument("--min-clinical-hits", type=int, default=1)
    p.add_argument("--allow-unknown-license", action="store_true", help="Disable license metadata filter; NOT recommended")
    p.add_argument("--output", default="artifacts/healthcare_tokenizer.json")
    a = p.parse_args(argv)
    if a.source == "local" and not a.local_jsonl:
        p.error("--local-jsonl required for local source")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    iterator = islice(healthcare_docs(source=a.source, dataset=a.dataset,
        dataset_config=a.dataset_config, split=a.split, local_jsonl=a.local_jsonl,
        partition="train", require_license=not a.allow_unknown_license,
        min_clinical_hits=a.min_clinical_hits), a.sample_docs)
    tokenizer.train_from_iterator(iterator)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(a.output)
    print(f"Healthcare tokenizer saved: {a.output}. ACTUAL vocabulary size: {tokenizer.get_vocab_size()}")
    print("Match model's vocab_size exactly; DO NOT change tokenizer between pretraining and full SFT.")


if __name__ == "__main__":
    main()
