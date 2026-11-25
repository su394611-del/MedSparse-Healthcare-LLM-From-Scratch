# Evaluation

The [November 20 baseline](#results-2026-11-20) evaluated the
[pretrained](https://huggingface.co/sudhanshu42844/medsparse-healthcare-pretrained-from-scratch) and
[Medical SFT](https://huggingface.co/sudhanshu42844/medsparse-healthcare-medical-sft) MedSparse checkpoints in the
[Model Playground](https://su394611-del.github.io/playground/)'s JavaScript engine. It also evaluated
[Tiny GPT](https://huggingface.co/sudhanshu42844/my-tiny-llm) separately on TinyStories.

The [November 25 update](#miriad-sft-evaluation-2026-11-25) adds MIRIAD SFT to the same browser benchmarks.
Medical Q&A perplexity fell from 253 to 59.6 versus Medical SFT (76% lower, paired 95% CI 71–81%).
The PubMed difference was inconclusive, and MedMCQA accuracy was 31.3% against a 38% majority-answer baseline.
In the separate 15-question PyTorch check, an informal review judged 3 MIRIAD SFT answers correct or mostly correct.

<!-- results:start -->
## Results (2026-11-20)

| Metric | Pretrained | Medical SFT |
|---|---|---|
| PubMed held-out perplexity (40 articles) | 572 (473–704) | 667 (546–823) |
| Medical Q&A held-out perplexity (80 answers) | 692 (643–741) | 253 (233–273) |
| Next-token top-1 accuracy, PubMed / Q&A | 15.6% / 14.4% | 14.3% / 21.6% |
| MedMCQA accuracy (150 questions, chance 25%) | 32.7% (25.7%–40.5%) | 28.7% (22.0%–36.4%) |
| Decode speed in the browser | 13.7 tokens/s | 14.0 tokens/s |

Tiny GPT ([my-tiny-llm](https://huggingface.co/sudhanshu42844/my-tiny-llm)): TinyStories validation perplexity 570 (552–592) on 100 stories; 204 tokens/s.

Brackets are 95% intervals (bootstrap over items for perplexity, Wilson for accuracy). Full numbers, row indices and sample generations: [`results.json`](results.json).
<!-- results:end -->

## MIRIAD SFT evaluation (2026-11-25)

[MIRIAD SFT](https://huggingface.co/sudhanshu42844/medsparse-healthcare-miriad-sft) (revision `b65a4702`) on the same
held-out items, scored by the same engine. Medical SFT was re-scored in the same run as a check and reproduced its
published numbers exactly (252.86 and 667.04).

| Metric | Pretrained | Medical SFT | MIRIAD SFT |
|---|---|---|---|
| Medical Q&A held-out perplexity (80 answers) | 692 (643–741) | 253 (233–273) | **59.6 (48.6–72.8)** |
| PubMed held-out perplexity (40 articles) | 572 (473–704) | 667 (546–823) | 586 (446–789) |
| Next-token top-1 accuracy, PubMed / Q&A | 15.6% / 14.4% | 14.3% / 21.6% | 16.8% / 33.8% |
| MedMCQA accuracy (150 questions, chance 25%, majority answer 38%) | 32.7% (25.7%–40.5%) | 28.7% (22.0%–36.4%) | 31.3% (24.5%–39.1%) |

Paired MIRIAD-vs-Medical-SFT perplexity ratio: Q&A 0.236 (0.194–0.287), PubMed 0.878 (0.71–1.08).

Also on 15 questions from MIRIAD papers held out of training (PyTorch, 192-token window): reference-answer
perplexity 9.5 vs 383 for Medical SFT; word-overlap F1 0.35 vs 0.19; an informal read-through judged 3 answers
correct or mostly correct, 5 vague and 7 wrong. Full numbers: [`results_miriad.json`](results_miriad.json).
Harness: `browser/driver_miriad.js` and `browser/aggregate_miriad.js`.


## What is measured

| Test | Data | How it is scored |
|---|---|---|
| PubMed perplexity | First 40 articles in `licensed_pubmed-0000.json.gz` (common-pile/pubmed_filtered) that pass the training filters and fall in the 2% validation partition | Packed pretraining window: first 129 tokens, no `<bos>`, 128 scored tokens per article |
| Medical Q&A perplexity | First 80 rows of medical-o1-reasoning-SFT (`en`) in the 5% validation bucket | `tokenize_example(seq_len=128)`: chat template, loss on answer tokens only, prompt trimmed from the left |
| MedMCQA | First 150 validation questions with four non-empty options | Each option scored by mean log-probability per token after the question (cloze), options not shown |
| TinyStories perplexity | First 100 stories of `TinyStories-valid.txt` | First 128 tokens, 127 scored |
| Expert routing | Held-out PubMed + Q&A tokens | Share of top-2 router selections per expert and layer |
| Samples, speed | First held-out questions; a fixed prompt | Greedy decoding; 64 tokens timed in one Web Worker |

The held-out selection re-implements `stable_partition` / `row_id` (`mixtral_lab/healthcare_corpus.py`) and
`bucket_for_question` (`mixtral_lab/medical_data.py`) in JavaScript (`browser/split.js`), including BLAKE2b,
and was checked against Python's `hashlib` on 308 strings, Unicode edge cases included (0 mismatches).

## Files

- `results.json`: aggregated results, model revisions, dataset row indices and sample generations.
- `browser/split.js`: held-out splits (BLAKE2b, casefold, whitespace rules matching Python).
- `browser/worker.js`: scoring on the playground engine (perplexity, cloze MCQ, routing counts, greedy samples, speed).
- `browser/driver.js`: selects the data, runs the workers in parallel and collects the raw scores.
- `browser/aggregate.js`: bootstrap and Wilson intervals, paired SFT-vs-pretrained comparison.
- `browser/test_node.js`: offline checks of the worker against a random-weight model in Node.
- `build_pages.py`, `training_chart.py`: render the evaluation section and the training chart into `index.html`.
- `tiny_log_page.py`: builds the Tiny GPT training-log page (`/training-logs/tiny-gpt/` on the main site).

## Re-run it

1. Open https://su394611-del.github.io/playground/worker-src.js in a desktop browser. The playground page itself now has a
   Content-Security-Policy that blocks `new Function` and blob workers; this plain-text file is on the same origin and
   has none. The evaluation downloads the 155 MB checkpoints.
2. Open the developer console and run:

```js
const base = 'https://raw.githubusercontent.com/su394611-del/MedSparse-Healthcare-LLM-From-Scratch/main/eval/browser/';
const src = async f => (await fetch(base + f)).text();
window.Split = new Function((await src('split.js')) + '\nreturn Split;')();
window.aggregateEval = new Function((await src('aggregate.js')) + '\nreturn aggregateEval;')();
window.__EV_WORKER_SRC = await src('worker.js');
new Function(await src('driver.js'))();
// Release 03 instead: window.aggregateMiriad = new Function((await src('aggregate_miriad.js')) + '\nreturn aggregateMiriad;')();
//                    new Function(await src('driver_miriad.js'))();  then aggregateMiriad(window.__EVAL)
// poll window.__EVAL.progress; when window.__EVAL.status === 'done':
// copy(JSON.stringify(aggregateEval(window.__EVAL), null, 2))
```

It takes about 15–20 minutes on an 8-thread laptop CPU. Perplexities should match `results.json` exactly for
the same model revisions; speeds depend on the machine.

## Training curve

The project page's **Training progress** chart currently shows held-out loss across the first two training stages
(untrained, after 5,000 pretraining steps, after 2,000 SFT steps), because per-step loss from those runs was
only printed to the console. The training scripts now also append every logged step to a JSON Lines file
(`MetricsLog` in `mixtral_lab/train_utils.py`):

- `scripts/pretrain_healthcare.py` writes `<out>/metrics.jsonl`: `train_ce` and `lr` every 10 steps, and
  `validation_ce` every `--eval-every` steps. Pretraining validation is off by default; pass `--eval-every 250`.
- `scripts/finetune_medical.py` writes `<out>/metrics.jsonl`: `train_completion_ce` and `lr` every 10 steps, and
  `val_completion_ce` every `--eval-every` steps (default 100).
- `--metrics-log PATH` overrides the location.

To publish the full curve after a run:

```bash
mkdir -p eval/training
cp checkpoints/healthcare_pretrained/metrics.jsonl eval/training/pretrain_metrics.jsonl
cp checkpoints/medical_sft/metrics.jsonl          eval/training/sft_metrics.jsonl
python eval/build_pages.py eval/results.json      # renders the per-step chart into index.html
```

If the new run produces new checkpoints, re-run the evaluation as well, so the held-out numbers match the published weights.

## Limits

These are language-modelling measurements on held-out text from a 38.8M-parameter model trained on
128-token windows. They do not measure clinical correctness or safety, and the models must not be used for
medical advice. The JavaScript engine is a float32 re-implementation of the published weights, not the
PyTorch training code.
