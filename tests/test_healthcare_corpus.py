import json
from pathlib import Path
from mixtral_lab.healthcare_corpus import (license_ok, medical_filter, row_id, stable_partition,
    healthcare_docs, extract_medical_text, HealthcarePackedTokens)


def test_license_filter():
    assert license_ok('Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/')
    assert license_ok('cc0-1.0')
    assert not license_ok('CC BY-NC-SA 4.0')
    assert not license_ok(None)
    assert not license_ok('All rights reserved')


def test_schema_and_clinical_filter():
    assert 'heart' in extract_medical_text({'title': 'Heart disease', 'abstract': 'Clinical cases'}, 'slinusc/PubMedAbstractsSubset').lower()
    assert medical_filter('Clinical patients underwent cardiac treatment.')
    assert not medical_filter('A butterfly was found on a meadow.')


def test_partition_stable():
    x = row_id({}, 'A clinical study of diabetes treatment')
    assert stable_partition(x, 42) == stable_partition(x, 42)


def test_only_medical_licensed_local_docs(tmp_path):
    path = tmp_path / 'sample.jsonl'
    rows = [
        {'text': 'A patient was treated for infection and fever.' * 4, 'license':'CC0-1.0'},
        {'text': 'The beach is beautiful and the sky blue.' * 4, 'license':'CC0-1.0'},
        {'text': 'In a clinical hospital, a patient received therapy.' * 4, 'license':'CC BY-NC-SA 4.0'},
    ]
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    docs = list(healthcare_docs(source='local', local_jsonl=str(path), validation_pct=0))
    assert len(docs) == 1
    assert 'patient' in docs[0]


def test_pretraining_tokens():
    data = HealthcarePackedTokens(tokenizer_path='data/byte_tokenizer.json', seq_len=16,
        source='local', local_jsonl='data/synthetic_healthcare_corpus.jsonl',
        validation_pct=0, repeat=False)
    import torch
    x = next(iter(data))
    assert x.shape == (17,) and x.dtype == torch.long


def test_medical_o1_capitalized_schema():
    from mixtral_lab.medical_data import format_example
    q, a = format_example({"Question": "What is hypertension?", "Response": "High blood pressure."},
                          schema="question_response")
    assert q == "What is hypertension?" and a == "High blood pressure."
