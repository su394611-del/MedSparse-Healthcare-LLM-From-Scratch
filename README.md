# Mixtral-style Healthcare Language Model — From-Scratch Pretraining and Full Medical SFT

A PyTorch research project implementing a **native Mixtral-style sparse Mixture-of-Experts (MoE) causal language model**. It trains a healthcare tokenizer, **pretrains randomly initialized model weights on biomedical text**, and then performs **full-parameter supervised fine-tuning (SFT)** on medical question–answer data.

This is a custom Mixtral-*style* implementation, **not** a checkpoint of Mistral AI's Mixtral model and not an assertion of weight or architecture compatibility with it. No externally pretrained language-model weights are required in the documented two-stage workflow.

> **Research use only.** Neither language-model loss nor medical SFT alone establishes clinical reliability. Do not use generated outputs for patient care without independent clinical validation, safeguards, and applicable regulatory review.

## Workflow at a glance

| Stage | Objective | Data | Training target | Output |
| --- | --- | --- | --- | --- |
| 0 | Audit corpus and train a healthcare BPE tokenizer | `common-pile/pubmed_filtered` | Sample up to 200,000 documents; target vocabulary 32,000 | `artifacts/healthcare_tokenizer.json` |
| 1 | Pretrain the native sparse-MoE model **from random initialization** | `common-pile/pubmed_filtered` | **5,000 optimizer steps** | `checkpoints/healthcare_pretrained_gpu_5000/last.pt` |
| 2 | **Full-parameter** medical SFT, initialized from Stage 1 | `FreedomIntelligence/medical-o1-reasoning-SFT` (`en`) | **2,000 optimizer steps** | `checkpoints/healthcare_medical_sft_gpu_2000/last.pt` |
| 3 | Evaluate held-out completion loss and clinical behavior | Held-out samples and expert-reviewed medical tests | Reporting and manual review | Evaluation results |

The documented single-device experiment initializes **38,779,008 parameters** (as reported by the SFT runtime) and updates all trainable weights. It is **not** the separate, larger `configs/healthcare_moe_1b.json` preset. Directory names are workflow labels, not proof that a run reached its target: inspect checkpoint metadata or training logs before reporting completed steps.

### Model architecture

The repository implements a Transformer with sparse MoE layers, top-k expert routing, RMSNorm, rotary position embeddings, grouped-query attention, and SwiGLU-style expert feed-forward blocks. The working experiment uses a reduced sparse-MoE architecture; preserve its **exact model configuration and tokenizer** across pretraining and SFT.

Other configuration presets may exist for larger parameter counts or different experimental contexts. They are **not** the trained configuration documented here, and a configured maximum context length is not evidence of effective long-context reasoning.

## Datasets and data responsibilities

| Stage | Dataset | Fields used | Notes |
| --- | --- | --- | --- |
| Pretraining | [`common-pile/pubmed_filtered`](https://huggingface.co/datasets/common-pile/pubmed_filtered) | Publication text and available metadata | Audit provenance, permissions, duplicates, and clinical relevance |
| Medical SFT | [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT), config `en` | `Question` and `Response` | The documented `question_response` formatter **does not train on** `Complex_CoT` |

The source datasets are streamed from Hugging Face. Their availability, terms, records, and metadata may change. Confirm the applicable licenses and rights for the exact data used; screen for personally identifiable or sensitive clinical content, duplicates, evaluation contamination, and unreliable medical assertions. Public hosting does not by itself establish permission for every downstream use.

## Setup

**Recommended interpreter:** Python 3.12. The commands below are written for **Windows PowerShell** and executed from the repository root. Use a virtual environment rather than installing packages into a global interpreter.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

Choose a supported PyTorch installation from the [official installation selector](https://pytorch.org/get-started/locally/) for the target runtime. `requirements.txt` declares Python packages but does **not** pin a specific accelerator wheel.

Verify the environment:

```powershell
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m scripts.pretrain_healthcare --help
python -m scripts.finetune_medical --help
```

### Select the configuration used to create the checkpoint

The original experiment's configuration filename was environment-specific. For public documentation, use a neutral alias **without changing its JSON contents**:

```powershell
$candidates = @(Get-ChildItem .\configs -File -Filter "healthcare_moe_gpu_*.json")
if ($candidates.Count -ne 1) {
    throw "Identify the exact configuration used to create the Stage 1 checkpoint, then copy it to configs/healthcare_moe_single.json."
}
Copy-Item $candidates[0].FullName .\configs\healthcare_moe_single.json
```

Commit `configs/healthcare_moe_single.json` alongside the updated documentation. The copy must be **byte-for-byte equivalent in architecture settings** to the config used for the pretrained checkpoint. Do not substitute `configs/healthcare_moe_1b.json` when loading weights produced by the smaller configuration.

## Stage 0 — Audit biomedical data and train the tokenizer

```powershell
python -m scripts.audit_healthcare_dataset `
    --dataset common-pile/pubmed_filtered `
    --max-rows 2000

python -m scripts.train_healthcare_tokenizer `
    --source hf `
    --dataset common-pile/pubmed_filtered `
    --sample-docs 200000 `
    --vocab-size 32000 `
    --output artifacts/healthcare_tokenizer.json
```

Validate the actual tokenizer vocabulary and special-token IDs **before** Stage 1:

```powershell
python -c "from tokenizers import Tokenizer; t=Tokenizer.from_file('artifacts/healthcare_tokenizer.json'); print('vocab:', t.get_vocab_size()); print({s: t.token_to_id(s) for s in ('<unk>','<bos>','<eos>','<pad>')})"
```

The tokenizer's actual vocabulary size must match `vocab_size` in the selected JSON model configuration. The tokenizer implementation must be able to encode the intended corpus, including unfamiliar characters. In particular, a BPE model configured with `unk_token='<unk>'` **must actually contain that token in its vocabulary**; otherwise encoding some inputs may fail even after many successful training steps. Reserve required special tokens during tokenizer training and test representative medical strings. **Do not add or reorder vocabulary IDs after pretraining without an explicit model-embedding migration.**

## Stage 1 — Pretraining from random weights (5,000 steps)

The pretraining target below is **5,000 optimizer steps**, not 5,000 tokens or a fixed number of passes through the dataset. With sequence length 128, batch size 1, and four gradient-accumulation microbatches, the nominal processing budget is **2,560,000 token positions**; actual non-padding token counts depend on the data pipeline.

```powershell
python -m scripts.pretrain_healthcare `
    --strategy single `
    --device cuda `
    --config configs/healthcare_moe_single.json `
    --tokenizer artifacts/healthcare_tokenizer.json `
    --source hf `
    --dataset common-pile/pubmed_filtered `
    --seq-len 128 `
    --batch-size 1 `
    --grad-accum 4 `
    --steps 5000 `
    --repeat `
    --lr 0.0003 `
    --warmup-steps 500 `
    --eval-every 1000 `
    --save-every 1000 `
    --out checkpoints/healthcare_pretrained_gpu_5000
```

For this workflow, `--strategy single` launches the script directly; **do not prefix it with `torchrun`**. The command intentionally omits `--bf16`. Confirm the final step and checkpoint contents rather than relying on the output directory name. For single-device runs, the checkpoint writer stores the latest checkpoint as `last.pt`.

```powershell
Test-Path .\checkpoints\healthcare_pretrained_gpu_5000\last.pt -PathType Leaf
Get-Content .\checkpoints\healthcare_pretrained_gpu_5000\training_manifest.json
```

**Checkpoint compatibility:** The Stage 2 model must use the same architecture, tokenizer file, vocabulary and token IDs as Stage 1. The from-scratch model is not directly load-compatible with an unrelated pretrained Mixtral checkpoint.

## Stage 2 — Full-parameter medical SFT (2,000 steps)

Start SFT from the **saved Stage 1 weights**, using a fresh optimizer. The training script reports whether all model weights are being updated; the observed SFT initialization reported `"update_all_weights": true`.

```powershell
python -m scripts.finetune_medical `
    --strategy single `
    --device cuda `
    --config configs/healthcare_moe_single.json `
    --tokenizer artifacts/healthcare_tokenizer.json `
    --pretrained checkpoints/healthcare_pretrained_gpu_5000/last.pt `
    --source hf `
    --dataset FreedomIntelligence/medical-o1-reasoning-SFT `
    --dataset-config en `
    --hf-split train `
    --schema question_response `
    --num-workers 0 `
    --seq-len 128 `
    --batch-size 1 `
    --grad-accum 4 `
    --steps 2000 `
    --lr 0.00002 `
    --warmup-steps 100 `
    --eval-every 200 `
    --save-every 500 `
    --out checkpoints/healthcare_medical_sft_gpu_2000
```

The `question_response` formatter accepts both lowercase and uppercase question/answer field names. Its loss masks the system and user prompt portion and supervises completion tokens. **Sequence length 128 may truncate medical explanations:** this is a reproducible experiment setting, not a recommendation for a clinical reasoning model.

### Resume SFT after interruption

Use `--resume` **instead of** `--pretrained` when continuing SFT. The script rejects using both together. First check that the SFT checkpoint exists and verify its recorded step:

```powershell
Test-Path .\checkpoints\healthcare_medical_sft_gpu_2000\last.pt -PathType Leaf
python -c "import torch; p=torch.load('checkpoints/healthcare_medical_sft_gpu_2000/last.pt', map_location='cpu', weights_only=True); print('step:', p.get('step', p.get('global_step', 'not recorded')))"
```

To continue toward the **2,000-step total target**, run the following complete command; it intentionally **does not include `--pretrained`**:

```powershell
python -m scripts.finetune_medical `
    --strategy single `
    --device cuda `
    --config configs/healthcare_moe_single.json `
    --tokenizer artifacts/healthcare_tokenizer.json `
    --resume checkpoints/healthcare_medical_sft_gpu_2000/last.pt `
    --source hf `
    --dataset FreedomIntelligence/medical-o1-reasoning-SFT `
    --dataset-config en `
    --hf-split train `
    --schema question_response `
    --num-workers 0 `
    --seq-len 128 `
    --batch-size 1 `
    --grad-accum 4 `
    --steps 2000 `
    --lr 0.00002 `
    --warmup-steps 100 `
    --eval-every 200 `
    --save-every 500 `
    --out checkpoints/healthcare_medical_sft_gpu_2000
```

Verify that the resumed logs continue at the expected optimizer step and that this script interprets `--steps` as the total target, not additional steps. Streaming dataset iteration may not resume from precisely the same example.

### Troubleshooting data and tokenization

- **`No rows for this distributed worker` in a single-process run:** check dataset access, `--hf-split train`, `--dataset-config en`, `--schema question_response`, and the loader's exception handling. Do not silently swallow all tokenization errors.
- **`Unk token <unk> not found in the vocabulary`:** the BPE model cannot encode at least one input. For an existing checkpoint, inspect the exact tokenizer and use narrowly scoped invalid-example handling as a temporary recovery measure. The robust long-term solution is a correctly trained tokenizer paired with correspondingly initialized model embeddings.
- **`--pretrained` and `--resume` together:** choose one. Use `--pretrained` for the first SFT run, `--resume` for continuation.
- **Checkpoint path:** the documented single-device writer uses `last.pt`, not an invented `step_XXXXXXXX` directory.

## Evaluation and reporting

The training scripts expose periodic next-token and completion-only validation cross-entropy/perplexity. These are **development metrics, not measurements of clinical correctness**. The observed SFT logs show improving held-out loss during training, but evaluation in that log covers a small supervised-token sample. Report the evaluation sample size and do not infer clinical capability from loss alone.

A release-quality evaluation should include an independent medical benchmark, expert review, hallucination and uncertainty tests, medication safety checks, data leakage/memorization checks, privacy review, and bias analysis. Track dataset versions, random seeds, full configuration, tokenizer hash, checkpoint step, and run metadata for reproducibility.

## Optional offline pipeline smoke test

```powershell
python -m scripts.two_stage_smoke --pretrain-steps 3 --sft-steps 3
```

This is a synthetic two-stage plumbing test. Passing it does **not** establish clinical performance, a completed large-scale pretraining run, or effective long-context behavior.

## Project layout

```text
mixtral_lab/model.py                   Native Transformer / sparse-MoE implementation
mixtral_lab/healthcare_corpus.py       Biomedical dataset streaming and filtering
mixtral_lab/medical_data.py           Medical SFT formatting and completion-only targets
scripts/audit_healthcare_dataset.py   Corpus inspection
scripts/train_healthcare_tokenizer.py BPE tokenizer training
scripts/pretrain_healthcare.py        Stage 1 next-token pretraining
scripts/finetune_medical.py           Stage 2 full-parameter SFT
scripts/two_stage_smoke.py            Offline pipeline check
configs/healthcare_moe_single.json    Exact architecture copy for the documented run
artifacts/healthcare_tokenizer.json   Healthcare tokenizer (generated)
checkpoints/                         Training output (generated)
```

### Publication and reproducibility note

This repository documents a **from-scratch, healthcare-domain research pipeline** and the commands used to train it. Published code and configurations are not a substitute for publishing verified checkpoint provenance, validation results, or dataset rights. Do not include generated checkpoints and sourced datasets in the repository without reviewing their contents, privacy implications, file sizes, and redistribution permissions.
This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0
