"""Quick source/schema/license/domain audit before billions of healthcare training tokens.
Does not output text or protected data; validation does not replace human audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from mixtral_lab.healthcare_corpus import (source_rows, extract_medical_text, license_ok,
    medical_filter, HF_PRETRAIN_CORPORA, DEFAULT_HF_CORPUS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['hf','local'], default='hf')
    p.add_argument('--dataset', default=DEFAULT_HF_CORPUS)
    p.add_argument('--dataset-config')
    p.add_argument('--split', default='train')
    p.add_argument('--local-jsonl')
    p.add_argument('--max-rows', type=int, default=1000)
    p.add_argument('--min-clinical-hits', type=int, default=1)
    a = p.parse_args()
    if a.max_rows <= 0:
        p.error('--max-rows must be positive')
    counts = Counter()
    for row in source_rows(a.source, a.dataset, a.dataset_config, a.split,
                           a.local_jsonl, 0, 42):
        counts['total_rows'] += 1
        if counts['total_rows'] > a.max_rows:
            break
        text = extract_medical_text(row, a.dataset if a.source=='hf' else 'local')
        if not text.strip():
            counts['invalid_text'] += 1
            continue
        metadata = row.get('metadata') or {}
        lic = metadata.get('license') if isinstance(metadata,dict) else None
        lic = lic or row.get('license')
        if a.source=='hf' and a.dataset=='slinusc/PubMedAbstractsSubset':
            lic = lic or 'cc0-1.0'  # dataset card only; audit underlying sources too
        if not license_ok(lic):
            counts['license_unknown_or_disallowed'] += 1
            continue
        if not medical_filter(text, min_hits=a.min_clinical_hits):
            counts['no_clinical_keyword'] += 1
            continue
        counts['passed_heuristic_filters'] += 1
    print(dict(counts))
    print('Manual rights, provenance, PHI, deduplication and clinical-quality checks still required.')


if __name__=='__main__':
    main()
