---
language: en
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
base_model: sudhanshu42844/medsparse-healthcare-pretrained-from-scratch
datasets:
  - FreedomIntelligence/medical-o1-reasoning-SFT
tags:
  - medsparse
  - mixtral-inspired
  - mixture-of-experts
  - sparse-moe
  - medical
  - healthcare
  - supervised-fine-tuning
  - full-fine-tuning
  - pytorch
---

# MedSparse Healthcare LLM — Full-Parameter Medical SFT

*Formerly **Mixtral Healthcare Medical SFT** (`sudhanshu42844/mixtral-healthcare-medical-sft`). The model and weights are unchanged; only the name changed, and the old URL redirects here.*

**Model type:** Custom PyTorch sparse Mixture-of-Experts causal language model with a Mixtral-inspired architecture. It does not contain weights from the official Mistral AI / Mixtral models.  
**Project:** [MedSparse-Healthcare-LLM-From-Scratch](https://github.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch)  
**Training stage:** Stage 2 — **full-parameter medical supervised fine-tuning**, initialized using the Stage 1 healthcare-pretrained native checkpoint.

> **Research only.** This model is not suitable for clinical decisions, diagnosis, medical treatment, patient triage, or medication dosing. Its medical correctness and safety have not been established.

## Lineage

1. Train a new MedSparse sparse MoE language model on healthcare-domain documents from **random initialization**.
2. Load the resulting **Stage 1 model weights** into the *same architecture*.
3. Initialize a **new AdamW optimizer** for SFT and update all model parameters, including embeddings, attention projections, routers, experts, layer norms, and the LM head.
4. Save a **separate Stage 2 checkpoint**, whose model weights are distinct from Stage 1.

**Base model:** [sudhanshu42844/medsparse-healthcare-pretrained-from-scratch](https://huggingface.co/sudhanshu42844/medsparse-healthcare-pretrained-from-scratch). No LoRA/QLoRA adapters or frozen base weights are used.

## Model details

| Attribute | Value |
|---|---|
| Total parameters | 38,779,008 |
| Transformer layers | 4 |
| Hidden dimension | 384 |
| Query / KV heads | 6 / 2 |
| Sparse experts | 4 per block; top 2 per token |
| Expert intermediate dimension | 768 |
| Configured maximum position length | 2048 tokens (not clinically validated at that length) |
| Medical SFT sequence length | 128 tokens |
| Microbatch / gradient accumulation | 1 / 4 |
| SFT learning rate | 2e-5 (configured run target) |
| SFT warmup | 100 optimizer steps (configured run target) |
| SFT evaluation cadence | 200 optimizer steps (configured run target) |
| SFT saving cadence | 500 optimizer steps (configured run target) |
| Optimizer steps in exported checkpoint | **2000** |
| Stage 2 target | 2,000 optimizer steps |
| Checkpoint status | **COMPLETE relative to configured step target** |

The training settings above describe the documented SFT run; only the saved optimizer step and model configuration are independently recoverable from the exported training checkpoint. For fully reproducible hyperparameter verification, preserve the original training command and logs privately.

## Medical instruction data and labeling

The documented SFT source is [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT), English configuration, using the `Question` and `Response` fields. The `Complex_CoT` field is not used as an SFT completion target.

Samples are formatted with `### System:`, `### User:`, and `### Assistant:` sections. The loss is **completion-only**: system and user prompt labels are masked with `-100`, so answer tokens (and the end-of-sequence token) determine the supervised cross-entropy. A MoE load-balancing objective is also added by the model.

Question-hash partitioning provides a held-out development validation split, but duplicate or semantically overlapping questions and upstream contamination still require auditing. The original medical dataset and upstream sources must be checked for licensing, privacy and factual integrity.

## Intended use and usage

- Research in full-weight medical instruction tuning and MoE specialization.
- Technical demonstrations of checkpoint lineage, custom transformer inference and evaluation.
- Offline evaluation using carefully reviewed, non-identifying medical examples.

Not approved for clinical care, patient communication without professional review, safety-critical triage, or any use implying regulatory clearance.

**This is not a Transformers `AutoModelForCausalLM` checkpoint.** Download the model and run the included native `load_model.py` to load `model.safetensors`, `config.json` and `tokenizer.json` using `mixtral_lab.MixtralLM` (the project's Python package keeps its original name).

```powershell
python -m pip install torch safetensors tokenizers huggingface_hub
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('sudhanshu42844/medsparse-healthcare-medical-sft'))"
```

From the downloaded directory:

```powershell
python load_model.py --sft --prompt "What does randomized controlled trial mean?"
```

## Evaluation and limitations

- Logged SFT validation completion-only cross-entropy and perplexity are **development metrics**, not measures of clinical competence.
- No independently verified clinical safety, accuracy, robustness, fairness, patient privacy or medication-safety benchmark is claimed.
- Fine-tuning data may contain synthetic answers, unsupported reasoning, errors or stale healthcare knowledge.
- Sequence-length and scale limitations apply; an architectural maximum is not proof of long-context ability.
- Hallucinations, overconfidence, omissions and potentially harmful responses remain possible.
- Verify human-reviewed clinical benchmarks, uncertainty handling, adversarial prompting, PHI leakage and memorization risks before considering any deployment beyond research.

## Provenance and licensing

- This repository contains a **separate complete SFT model state**, not merely a fine-tuning adapter.
- The SFT checkpoint is linked to the Stage 1 base model for lineage purposes, not for automatic runtime loading.
- Weights are exported without optimizer state, local paths, logs, or private datasets.
- The model weights and code in this repository are released under the **Apache License 2.0** (Copyright 2026 Sudhanshu Shukla); see [`LICENSE`](LICENSE). The training data keeps its own terms: check [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) and its upstream sources for any obligations that apply to your use.
