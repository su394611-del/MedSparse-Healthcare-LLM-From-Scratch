---
language: en
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
datasets:
  - common-pile/pubmed_filtered
tags:
  - medsparse
  - mixtral-inspired
  - mixture-of-experts
  - sparse-moe
  - from-scratch
  - healthcare
  - biomedical
  - pytorch
  - language-modeling
---

# MedSparse Healthcare Language Model — Pretrained from Scratch

*Formerly **Mixtral Healthcare Pretrained** (`sudhanshu42844/mixtral-healthcare-pretrained-from-scratch`). The model and weights are unchanged; only the name changed, and the old URL redirects here.*

**Model type:** Custom PyTorch decoder-only language model with sparse Mixture-of-Experts (MoE) feed-forward layers and a Mixtral-inspired architecture.  
**Project:** [MedSparse-Healthcare-LLM-From-Scratch](https://github.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch)  
**Training stage:** Stage 1 — healthcare-domain autoregressive pretraining from random initialization.

> **Research model; not a clinical device or medical decision-support system.** MedSparse Healthcare is a custom Mixtral-**inspired** implementation and does not contain weights from the official Mistral/Mixtral models.

## Model details

| Attribute | Value |
|---|---|
| Model architecture | Decoder-only Transformer with top-2 sparse MoE |
| Model parameters | 38,779,008 total (all trainable during pretraining) |
| Transformer decoder layers | 4 |
| Hidden dimension | 384 |
| Query attention heads | 6 |
| Key/value attention heads | 2 (grouped-query attention) |
| Feed-forward experts per block | 4; top 2 selected per token |
| Expert intermediate dimension | 768 |
| Position encoding / normalization | RoPE / RMSNorm |
| Expert nonlinearity | SwiGLU |
| Configured maximum position length | 2048 tokens (not a demonstrated effective long-context capability) |
| Pretraining sequence length | 128 tokens (training-run setting; confirm using original run logs) |
| Optimizer steps in exported checkpoint | **5000** |
| Stage 1 target | 5,000 optimizer steps |
| Checkpoint status | **COMPLETE relative to configured step target** |
| Output formats | `model.safetensors`, `config.json`, `tokenizer.json` |

The architecture configuration and tokenizer shipped alongside the weights are authoritative for reconstruction. The vocabulary size must match the tokenizer exactly.

## Training data and objective

The documented pretraining source is [common-pile/pubmed_filtered](https://huggingface.co/datasets/common-pile/pubmed_filtered), screened by the repository's healthcare text and license-metadata filters. Confirm the exact data selection and licensing against the run manifest and underlying datasets before broader distribution. No general-web or official pretrained Mixtral weights are intentionally loaded in this training stage.

Training optimizes causal next-token cross-entropy and a weighted MoE router load-balancing auxiliary objective. Every model parameter is eligible for updates. The workflow uses a dedicated healthcare tokenizer trained from domain text. A deterministic document-hash partition is used for development validation; its exact composition depends on training configuration.

Dataset sources may contain errors, duplicated content, historical medical guidance, embedded personal information, or licensing ambiguity. Filtering metadata is not independent confirmation of permission to redistribute learned weights.

## Intended uses

- Research into sparse expert routing and decoder-only language modeling.
- Reproducible healthcare-domain language-model pretraining experiments.
- Initialization of a **separate full-parameter medical SFT model**: [MedSparse Healthcare Medical SFT](https://huggingface.co/sudhanshu42844/medsparse-healthcare-medical-sft).

**Not intended for:** diagnosis, treatment recommendations, medication dosing, patient triage, professional medical advice, automated patient decisions, or deployment without expert-led validation.

## Loading the model

This is a **native custom PyTorch architecture**, not a registered Hugging Face Transformers `AutoModelForCausalLM` format. Download the repository and use the included `load_model.py`, which imports `mixtral_lab.MixtralLM` (the project's Python package keeps its original name).

```powershell
python -m pip install torch safetensors tokenizers huggingface_hub
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('sudhanshu42844/medsparse-healthcare-pretrained-from-scratch'))"
```

Then run `python load_model.py --prompt "Clinical research explores"` from within the downloaded model directory. Generation is a research demo, not medical validation.

## Evaluation and limitations

- Training cross-entropy and perplexity assess language-model loss, not medical accuracy.
- **No independently verified clinical benchmarks, medical safety evaluation, or hallucination guarantees are reported for this release.**
- Architecturally supported context length does not imply demonstrated generalization to that length; pretraining used short sequences.
- Results depend on the source snapshot, tokenizer, token filtering, optimizer settings and validation partition.
- This model is not interchangeable with official Mixtral checkpoints.

## Provenance and licensing

- Checkpoint tensors are exported without optimizer state or private training paths.
- The model architecture source is available in the linked GitHub repository.
- The model weights and code in this repository are released under the **Apache License 2.0** (Copyright 2026 Sudhanshu Shukla); see [`LICENSE`](LICENSE). The training data keeps its own terms: check [common-pile/pubmed_filtered](https://huggingface.co/datasets/common-pile/pubmed_filtered) and its upstream sources for any obligations that apply to your use.
- Generated model metadata in `release_metadata.json` includes the saved optimizer step and configuration, but not a verifiable medical-safety assessment.
