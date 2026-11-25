# MedSparse Healthcare: pretrain FROM SCRATCH, then FULL fine-tune

*MedSparse Healthcare (formerly Mixtral Healthcare) is a custom sparse Mixture-of-Experts language model with a Mixtral-inspired architecture. It does not use or distribute official Mistral AI or Mixtral weights.*

**Released work:** the 38.8M-parameter [pretrained](https://huggingface.co/sudhanshu42844/medsparse-healthcare-pretrained-from-scratch), [medical SFT](https://huggingface.co/sudhanshu42844/medsparse-healthcare-medical-sft), and [MIRIAD SFT](https://huggingface.co/sudhanshu42844/medsparse-healthcare-miriad-sft) checkpoints on Hugging Face, the [technical report](https://su394611-del.github.io/research/medsparse-healthcare-2026/) (DOI [10.5281/zenodo.24461541](https://doi.org/10.5281/zenodo.24461541)), the [project site](https://su394611-del.github.io/MedSparse-Healthcare-LLM-From-Scratch/) with an in-browser playground, and the [engineering wiki](https://github.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch/wiki). Held-out results are in [`eval/results.json`](eval/results.json) and [`eval/results_miriad.json`](eval/results_miriad.json).

## Run MedSparse

- **Use a released 38.8M checkpoint.** Download the complete pretrained, medical SFT, or MIRIAD SFT bundle and follow its model card. The native loader needs the matching `model.safetensors`, `config.json`, and `tokenizer.json` from that bundle.
- **Inspect the released architecture.** Run `python -m scripts.inspect_model --config configs/healthcare_moe_single.json`. The inspector reports 38,779,008 parameters.
- **Exercise both training stages on a CPU.** Run `python -m scripts.two_stage_smoke --pretrain-steps 1 --sft-steps 1` with the bundled synthetic fixtures.

See [the run guide](RUN_MEDSPARSE.md) for setup and required files, or open [Getting Started](https://github.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch/wiki/Getting-Started) and [Repository Map](https://github.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch/wiki/Repository-Map) in the wiki. The 1.08B training commands below use a separate architecture preset.

This repository extends the original PyTorch **native, randomly initialized sparse-MoE architecture (Mixtral-inspired)**. Its intended sequence is:

1. **Build a tokenizer only on biomedical / healthcare publications from Hugging Face.**
2. **Stage 1 — pretrain every model weight from RANDOM INITIALIZATION** on an auditable healthcare text corpus with standard next-token loss and auxiliary MoE load-balancing loss. **No Mistral/Mixtral pretrained weights are downloaded or used.**
3. **Stage 2 — use the Stage 1 native checkpoint as initialization and fully fine-tune the same model** on separate medical instruction/QA data from Hugging Face. All weights remain trainable; a *new* optimizer is initialized.
4. **Stage 3 — continue full-parameter SFT from the Stage 2 checkpoint** using MIRIAD research Q&A and a paper-level validation split.
5. Evaluate on held-out data.

The **code is NOT a pretrained billion-parameter healthcare model**. The CPU-only smoke test proves the pipeline, not medical competence. Running 1B/46.7B training, especially at large contexts, requires substantial CUDA GPU resources.

## Data — Hugging Face, healthcare/biomedical only

| Stage | HF dataset | Schema | Important caveat |
|---|---|---|---|
| 1, default | `common-pile/pubmed_filtered` | `text`, `metadata.license` | ~4.77M PubMed Central open-access articles; enforce recognized CC0/CC-BY metadata AND clinical-keyword heuristic; manual audit required |
| 1, optional | `slinusc/PubMedAbstractsSubset` | `title`, `abstract` | ~2.39M PubMed abstracts; **dataset-level** CC0 metadata does not prove every original article's rights |
| 2, preferred | `FreedomIntelligence/medical-o1-reasoning-SFT`, config `en` | `Question`, `Response` | ~19.7k English medical examples, Apache-2.0 dataset card; ignores `Complex_CoT` |
| 2, alternative | `huzaifa525/Medical_Intelligence_Dataset_76k_2026_Edition` | `input`, `output` | ~76k medical QA rows, Apache-2.0 dataset card; audit third-party upstream sources and PHI |

Sources: https://huggingface.co/datasets/common-pile/pubmed_filtered ; https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT .

*Being on Hugging Face is not permission to train on all data.* License, origin, patient privacy, synthetic-data integrity, duplicates, contamination and clinical relevance need independent review. Only the two above **allowlisted healthcare pretraining sources** are accepted by the new Stage 1 streaming loader; there is no general-web fallback. The keyword filter is a preliminary screen, NOT a definitive clinical-text classifier.

### Requirements

Python >=3.10; PyTorch >=2.6, `datasets`, `tokenizers`, `pytest`. Install on an appropriately provisioned CUDA machine:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

For CUDA, install the appropriate **CUDA-enabled PyTorch wheel** for your machine per https://pytorch.org/get-started/locally/. Check `python -c 'import torch; print(torch.cuda.is_available())'`.

### Stage 0 — corpus audit and healthcare tokenizer

```bash
python -m scripts.audit_healthcare_dataset \
  --dataset common-pile/pubmed_filtered --max-rows 2000

python -m scripts.train_healthcare_tokenizer \
  --source hf --dataset common-pile/pubmed_filtered \
  --sample-docs 200000 --vocab-size 32000 \
  --output artifacts/healthcare_tokenizer.json
```

**Verify the printed ACTUAL tokenizer vocabulary size is precisely `32000`.** If not, change `vocab_size` in the model config to match *before Stage 1*. Keep the same tokenizer for both stages; mismatched token IDs or changing vocabulary will invalidate the Stage 1 model.

### Stage 1 — pretrain the 1.08B MedSparse MoE from random initialization

Example **8×CUDA GPU cluster** command; only a template and not benchmarked at full scale:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.pretrain_healthcare \
  --strategy fsdp2 \
  --config configs/healthcare_moe_1b.json \
  --tokenizer artifacts/healthcare_tokenizer.json \
  --source hf --dataset common-pile/pubmed_filtered \
  --seq-len 2048 --batch-size 1 --grad-accum 8 \
  --target-tokens 10000000000 --repeat --bf16 \
  --lr 0.0003 --warmup-steps 1000 \
  --eval-every 1000 --save-every 2000 \
  --out checkpoints/healthcare_pretrained_1b
```

- `--target-tokens 10000000000` means **10B processed training tokens**, NOT 10B model parameters. The *number of model parameters* comes only from the JSON architecture.
- The 10B token target may cycle the PubMed corpus using `--repeat`, and is **not** 10B *unique* tokens. Exact streaming order on resume is not preserved; complete rigorous reproducibility needs dataset cursor checkpointing and provenance logs.
- Validation is a held-out deterministic hash partition (2% default) of pretraining documents. Exact text duplicates hash to the same partition; semantic near-duplicates need a separate dedup process.
- FSDP2 CUDA functionality is an **experimental cluster template**. Expert parallelism, tensor/pipeline parallelism and context parallelism are not implemented in this project.

Stage 1 output is `checkpoints/healthcare_pretrained_1b/step_XXXXXXXX/` for FSDP2, or `checkpoints/healthcare_pretrained_1b/last.pt` for single-GPU. Do not use Hugging Face's pretrained Mixtral weights here: the native custom architecture is not weight-compatible.

### Stage 2 — FULL parameter medical SFT (from Stage 1)

Every trainable parameter is optimized (including embeddings, attention, router, all experts); no LoRA, QLoRA, frozen base or external pretrained weights.

**If Stage 1 used FSDP2, point `--pretrained` at its final `step_XXXXXXXX` directory**:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.finetune_medical \
  --strategy fsdp2 \
  --config configs/healthcare_moe_1b.json \
  --tokenizer artifacts/healthcare_tokenizer.json \
  --pretrained checkpoints/healthcare_pretrained_1b/step_XXXXXXXX \
  --source hf --dataset FreedomIntelligence/medical-o1-reasoning-SFT \
  --dataset-config en --schema question_response \
  --seq-len 2048 --batch-size 1 --grad-accum 8 \
  --steps 3000 --lr 0.00002 --warmup-steps 100 \
  --eval-every 200 --save-every 500 --bf16 \
  --out checkpoints/healthcare_medical_sft_1b
```

The `question_response` loader supports **uppercase `Question`/`Response`**, matching the English medical-o1 corpus. It **excludes `Complex_CoT`** rather than teaching verbose generated reasoning. SFT masks all system/user prompt tokens from the loss, supervising only answer tokens and EOS. `--pretrained` loads *weights only* so the SFT optimizer is new.

**If Stage 1 was a single-device run**, use `--strategy single` for Stage 2 and `--pretrained checkpoints/healthcare_pretrained_1b/last.pt`. The same `config` and `tokenizer` must be used in both stages.

Alternative medical SFT dataset:

```bash
--dataset huzaifa525/Medical_Intelligence_Dataset_76k_2026_Edition --schema input_output
```

Review its upstream data lineage and patient privacy first. Do not publish automatically.

### Stage 3 — MIRIAD SFT

The [released MIRIAD checkpoint](https://huggingface.co/sudhanshu42844/medsparse-healthcare-miriad-sft) continues Stage 2 training. It was saved at 1,507,000 optimizer steps with 6,031,000 processed training examples, including repeats. A resume issue affected the shuffled stream, so unique row coverage is unverified. See Section 7 of the [technical report](https://su394611-del.github.io/research/medsparse-healthcare-2026/) for the coverage analysis.

### Evaluation

The pretraining script can calculate next-token validation cross-entropy / perplexity by setting `--eval-every`. Stage 2 calculates held-out answer-only validation loss and perplexity by setting `--eval-every` and `--eval-batches`. Both are **development metrics**, not medical accuracy or patient safety. Add expert-reviewed testing for medical questions, uncertainty, drug safety, patient triage, bias, PHI exposure, memorization and hallucinations prior to any real clinical use.

### Full offline two-stage smoke test (NO Hugging Face access needed)

```bash
python -m scripts.two_stage_smoke --pretrain-steps 3 --sft-steps 3
```

This uses the bundled 260-token byte tokenizer and authored, synthetic clinical passages and QA. It creates:

```text
checkpoints/two_stage_smoke/
  stage1_pretrained/last.pt
  stage1_pretrained/training_manifest.json
  stage2_fully_finetuned/last.pt
  smoke_manifest.json
```

It explicitly compares Stage 1 and Stage 2 weight tensors and fails if no weights change. The test model has **548,832** parameters—not a billion. The sample passages have not been downloaded from HF. Do not interpret generated text from this model as medical evidence.

## Architecture and million-token context reality

- `configs/healthcare_moe_1b.json`: ~1.08B total parameters; 8 experts, top-2 routing.
- `configs/healthcare_mixtral_8x7b.json`: ~46.7B total parameters; ~12.9B structurally active parameters/token.
- `configs/healthcare_million_sliding_experimental.json`: max position index 1,048,576 with **4,096-token sliding attention**. This **does not** give global 1-million-token information access. One-million-token *full* attention or robust million-token reasoning is **not implemented or validated**.
- Optimizer training states with AdamW commonly require roughly 16 bytes/parameter or more before activation memory, depending on optimizer/precision. For 46.7B, that is at least ~747 GB aggregate, before activation/KV, fragmentation, and communication. You need a sizeable distributed GPU cluster and expert parallelism for efficient training.
- A genuine 1M-token globally usable context would require trained long-context attention (distributed ring/context parallelism, memory hierarchy or advanced attention), curated long clinical documents, and long-context retrieval/needle-in-haystack evaluation. No such result is claimed.

## Files

```text
mixtral_lab/model.py                  Native sparse MoE with RMSNorm, RoPE, GQA, SwiGLU
mixtral_lab/healthcare_corpus.py      HF healthcare allowlist, license/clinical filters, streaming
mixtral_lab/medical_data.py          Medical instruction formatting and answer-only masking
scripts/audit_healthcare_dataset.py  Schema and metadata audit before downloading terabytes
scripts/train_healthcare_tokenizer.py Healthcare BPE tokenizer from scratch
scripts/pretrain_healthcare.py       STAGE 1 random initialization, all parameters updated
scripts/finetune_medical.py          STAGE 2 full SFT loading Stage 1 native checkpoint
scripts/two_stage_smoke.py           Offline two-stage train/load/weight-change proof
scripts/generate.py, evaluate.py     Text generation, general perplexity reference
configs/healthcare_*.json           1B, 46.7B, million-position experimental presets
data/                                Byte tokenizer and synthetic fixtures for the offline smoke test and unit tests
```

**Status as packaged:** The small-model two-stage CPU smoke passed, and the automated tests in `tests/` passed. Live HF dataset loading, BPE tokenizer dependency, multi-GPU FSDP2 training, billion-parameter training, and full 1M context functionality have NOT been executed here. The packaging environment had no CUDA device or Hugging Face access, and `tokenizers` is only available after you install the requirements. This is a runnable, research-grade training codebase, not production software.

This project is licensed under the [Apache License, Version 2.0](LICENSE). © 2026 Sudhanshu Shukla.
