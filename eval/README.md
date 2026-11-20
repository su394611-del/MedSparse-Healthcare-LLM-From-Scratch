# Evaluation

Held-out evaluation of the two published checkpoints
([pretrained](https://huggingface.co/sudhanshu42844/mixtral-healthcare-pretrained-from-scratch),
[medical SFT](https://huggingface.co/sudhanshu42844/mixtral-healthcare-medical-sft)) and the
[Tiny GPT](https://huggingface.co/sudhanshu42844/my-tiny-llm), run in a web browser on the same JavaScript
engine as the [Model Playground](https://su394611-del.github.io/playground/).

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

## Re-run it

1. Open https://su394611-del.github.io/playground/ in a desktop browser (the evaluation downloads both 155 MB checkpoints).
2. Open the developer console and run:

```js
const base = 'https://raw.githubusercontent.com/su394611-del/Mixtral-Healthcare-LLM-From-Scratch/main/eval/browser/';
const src = async f => (await fetch(base + f)).text();
window.Split = new Function((await src('split.js')) + '\nreturn Split;')();
window.aggregateEval = new Function((await src('aggregate.js')) + '\nreturn aggregateEval;')();
window.__EV_WORKER_SRC = await src('worker.js');
new Function(await src('driver.js'))();
// poll window.__EVAL.progress; when window.__EVAL.status === 'done':
// copy(JSON.stringify(aggregateEval(window.__EVAL), null, 2))
```

It takes about 15–20 minutes on an 8-thread laptop CPU. Perplexities should match `results.json` exactly for
the same model revisions; speeds depend on the machine.

## Limits

These are language-modelling measurements on held-out text from a 38.8M-parameter model trained on
128-token windows. They do not measure clinical correctness or safety, and the models must not be used for
medical advice. The JavaScript engine is a float32 re-implementation of the published weights, not the
PyTorch training code.
