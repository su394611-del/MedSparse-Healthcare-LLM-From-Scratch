# Run MedSparse Healthcare

## Choose a workflow

| Goal | Entry point and required files |
| --- | --- |
| Run the released 38.8M checkpoint | Download the matching model bundle. Use its native `load_model.py`, `model.safetensors`, `config.json`, and `tokenizer.json`. See the [pretrained model card](model_cards/medsparse-healthcare-pretrained-from-scratch.md), [medical SFT model card](model_cards/medsparse-healthcare-medical-sft.md), or [MIRIAD SFT model card](https://huggingface.co/sudhanshu42844/medsparse-healthcare-miriad-sft). |
| Inspect the 38.8M architecture | Use [scripts/inspect_model.py](scripts/inspect_model.py) with [configs/healthcare_moe_single.json](configs/healthcare_moe_single.json). |
| Verify both training stages offline | Run [scripts/two_stage_smoke.py](scripts/two_stage_smoke.py). It uses [configs/smoke.json](configs/smoke.json), the [byte tokenizer](data/byte_tokenizer.json), [synthetic pretraining data](data/synthetic_healthcare_corpus.jsonl), and [synthetic medical Q&A](data/medical_demo.jsonl). |
| Train with your own data | Use [scripts/pretrain_healthcare.py](scripts/pretrain_healthcare.py) and [scripts/finetune_medical.py](scripts/finetune_medical.py). Both use [mixtral_lab/](mixtral_lab/) and require a tokenizer whose vocabulary matches the selected model config. The [healthcare tokenizer](artifacts/healthcare_tokenizer.json) is tracked here. |

## Verify a checkout

From the repository root in PowerShell with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.inspect_model --config configs/healthcare_moe_single.json
.\.venv\Scripts\python.exe -m scripts.two_stage_smoke --pretrain-steps 1 --sft-steps 1
```

After dependencies are installed, the smoke run uses the CPU and bundled synthetic data. It writes `checkpoints/two_stage_smoke/smoke_manifest.json` and verifies that full fine-tuning changed weight tensors from the pretraining checkpoint. The smoke config is smaller than the released model. The 1.08B config in [README.md](README.md) is a separate training target.

Keep local checkpoints, downloaded datasets, and virtual environments outside version control.
