"""CPU-only offline proof: pretrain random Mixtral -> full medical SFT -> verify weight changes.

The data are small authored synthetic demonstrations, NOT Hugging Face downloads.
It proves that the software path runs, not that a useful medical model was trained.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='checkpoints/two_stage_smoke')
    p.add_argument('--pretrain-steps', type=int, default=3)
    p.add_argument('--sft-steps', type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = out / 'stage1_pretrained'
    sft = out / 'stage2_fully_finetuned'
    py = sys.executable
    pretrain = [py, '-m', 'scripts.pretrain_healthcare',
        '--config', 'configs/smoke.json', '--device', 'cpu', '--tokenizer', 'data/byte_tokenizer.json',
        '--source', 'local', '--local-jsonl', 'data/synthetic_healthcare_corpus.jsonl',
        '--seq-len', '32', '--batch-size', '1', '--grad-accum', '1',
        '--steps', str(a.pretrain_steps), '--warmup-steps', '1', '--repeat',
        '--eval-every', '0', '--out', str(pre)]
    finetune = [py, '-m', 'scripts.finetune_medical',
        '--config', 'configs/smoke.json', '--device', 'cpu', '--tokenizer', 'data/byte_tokenizer.json',
        '--pretrained', str(pre / 'last.pt'), '--source', 'local',
        '--local-jsonl', 'data/medical_demo.jsonl', '--seq-len', '32',
        '--batch-size', '1', '--grad-accum', '1', '--steps', str(a.sft_steps),
        '--warmup-steps', '1', '--eval-every', '0', '--save-every', '0',
        '--out', str(sft)]
    for cmd in (pretrain, finetune):
        subprocess.run(cmd, check=True)
    pre_weights = torch.load(pre / 'last.pt', map_location='cpu', weights_only=True)['model']
    sft_weights = torch.load(sft / 'last.pt', map_location='cpu', weights_only=True)['model']
    changed = [k for k in pre_weights if not torch.equal(pre_weights[k], sft_weights[k])]
    if not changed:
        raise AssertionError('Stage 2 did not update pretrained weights!')
    manifest = {'stage1_completed': True, 'stage2_completed': True,
                'pretrained_from_random_init': True,
                'stage1_data': 'bundled synthetic healthcare passages, NOT HF',
                'stage2_data': 'bundled synthetic medical QA, NOT HF',
                'stage1_optimizer_steps': a.pretrain_steps,
                'stage2_optimizer_steps': a.sft_steps,
                'changed_parameter_tensors': len(changed),
                'stage1_checkpoint': str(pre/'last.pt'),
                'stage2_checkpoint': str(sft/'last.pt'),
                'note': 'Smoketest architecture only. Not a trained clinically useful model.'}
    (out/'smoke_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
