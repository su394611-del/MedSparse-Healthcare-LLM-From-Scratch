/*
 * Aggregates window.__EVAL.raw into the published summary (results.json).
 * Perplexity CIs: 2,000-sample bootstrap over evaluation items (documents / questions / stories),
 * token-weighted, fixed seed. Accuracy CIs: Wilson score interval. Paired SFT-vs-Pretrained
 * difference: paired bootstrap over the same items.
 */
function aggregateEval(S) {
  'use strict';
  function rng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function pct(a, p) { var s = a.slice().sort(function (x, y) { return x - y; }); var i = (s.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return s[lo] + (s[hi] - s[lo]) * (i - lo); }
  function r(x, d) { var k = Math.pow(10, d == null ? 3 : d); return Math.round(x * k) / k; }
  var B = 2000;

  function lm(items, vocab) {
    var nll = 0, n = 0, top1 = 0;
    items.forEach(function (it) { nll += it.nll; n += it.n; top1 += it.top1; });
    var R = rng(42), boots = [];
    for (var b = 0; b < B; b++) {
      var bn = 0, bt = 0;
      for (var i = 0; i < items.length; i++) { var it = items[Math.floor(R() * items.length)]; bn += it.nll; bt += it.n; }
      boots.push(Math.exp(bn / bt));
    }
    var mean = nll / n;
    return { items: items.length, tokens: n, meanNll: r(mean, 4), perplexity: r(Math.exp(mean), 2), ci95: [r(pct(boots, 0.025), 2), r(pct(boots, 0.975), 2)],
             bitsPerToken: r(mean / Math.LN2, 3), top1Accuracy: r(top1 / n, 4), uniformPerplexity: vocab };
  }
  function paired(a, b) {   // ratio of perplexities b / a on the same items, paired bootstrap
    var R = rng(7), boots = [];
    for (var k = 0; k < B; k++) {
      var na = 0, ta = 0, nb = 0, tb = 0;
      for (var i = 0; i < a.length; i++) { var j = Math.floor(R() * a.length); na += a[j].nll; ta += a[j].n; nb += b[j].nll; tb += b[j].n; }
      boots.push(Math.exp(nb / tb - na / ta));
    }
    var sa = 0, ta2 = 0, sb = 0, tb2 = 0;
    a.forEach(function (x) { sa += x.nll; ta2 += x.n; }); b.forEach(function (x) { sb += x.nll; tb2 += x.n; });
    return { ratio: r(Math.exp(sb / tb2 - sa / ta2), 3), ci95: [r(pct(boots, 0.025), 3), r(pct(boots, 0.975), 3)] };
  }
  function wilson(k, n) { var z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [r(c - h, 3), r(c + h, 3)]; }
  function mcq(items) {
    var kN = 0, kS = 0, dist = [0, 0, 0, 0];
    items.forEach(function (it) { if (it.predNorm === it.cop) kN++; if (it.predSum === it.cop) kS++; dist[it.predNorm]++; });
    var gold = [0, 0, 0, 0]; items.forEach(function (it) { gold[it.cop]++; });
    var majority = Math.max.apply(null, gold) / items.length;
    return { items: items.length, accuracy: r(kN / items.length, 3), ci95: wilson(kN, items.length), accuracyUnnormalized: r(kS / items.length, 3),
             ci95Unnormalized: wilson(kS, items.length), chance: 0.25, majorityClassBaseline: r(majority, 3),
             predictedLetterShare: dist.map(function (d) { return r(d / items.length, 3); }), goldLetterShare: gold.map(function (g) { return r(g / items.length, 3); }) };
  }
  function routing(route) {
    if (!route) return null;
    return route.map(function (layer) {
      var tot = layer.reduce(function (a, b) { return a + b; }, 0), H = 0;
      var share = layer.map(function (c) { var p = c / tot; if (p > 0) H -= p * Math.log(p); return r(p, 3); });
      return { share: share, normalizedEntropy: r(H / Math.log(layer.length), 3) };
    });
  }
  function sumRoute(a, b) { if (!a) return b; return a.map(function (l, i) { return l.map(function (c, e) { return c + b[i][e]; }); }); }

  var R = S.raw;
  var preMcq = R.pre_mcq_a.items.concat(R.pre_mcq_b.items), sftMcq = R.sft_mcq_a.items.concat(R.sft_mcq_b.items);
  var bothRight = 0; preMcq.forEach(function (p, i) { if (p.predNorm === p.cop && sftMcq[i].predNorm === sftMcq[i].cop) bothRight++; });
  var qaCut = R.sft_qa.items.filter(function (x) { return x.questionCut; }).length;
  var out = {
    generated: S.finished, harnessCommit: window.__EV_COMMIT || null,
    environment: { browser: (navigator.userAgentData && navigator.userAgentData.brands || []).map(function (b) { return b.brand + ' ' + b.version; }).join(', ') || navigator.userAgent, cpuThreads: navigator.hardwareConcurrency, engine: 'Model Playground JavaScript engine (float32, CPU, Web Worker)' },
    models: {
      pretrained: { repo: 'sudhanshu42844/medsparse-healthcare-pretrained-from-scratch', revision: S.shas.pre, parameters: 38779008 },
      sft: { repo: 'sudhanshu42844/medsparse-healthcare-medical-sft', revision: S.shas.sft, parameters: 38779008 },
      tiny: { repo: 'sudhanshu42844/my-tiny-llm', revision: S.shas.tiny, parameters: 1060992 }
    },
    data: {
      pubmed: { source: 'common-pile/pubmed_filtered, first original shard licensed_pubmed-0000.json.gz, file order', rule: 'training filters (clinical keyword, CC0/CC-BY license) + 2% validation partition from healthcare_corpus.stable_partition, seed 42', format: 'packed pretraining window: first 129 tokens, no <bos>, 128 scored', selected: S.data.pubmedInfo.selected, scanned: S.data.pubmedInfo.scanned, rejected: S.data.pubmedInfo.rejected, rowIndices: S.data.ids.pubmed },
      qa: { source: 'FreedomIntelligence/medical-o1-reasoning-SFT, config en (medical_o1_sft.json, file order; Question/Response)', rule: '5% validation bucket from medical_data.bucket_for_question, seed 42', format: 'tokenize_example(seq_len=128): SFT chat template, completion-only scoring, prompt kept from the right', selected: S.data.qaInfo.selected, scanned: S.data.qaInfo.scanned, questionFullyKeptShare: r(1 - qaCut / R.sft_qa.items.length, 3), rowIndices: S.data.ids.qa },
      mcq: { source: 'openlifescienceai/medmcqa, validation split, first 150 rows with four non-empty options', format: 'SFT chat template; each option scored by mean log-probability per token (cloze); no options shown in the prompt', rowIndices: S.data.ids.mcq },
      tiny: { source: 'roneneldan/TinyStories, TinyStories-valid.txt, first 100 stories', format: 'first 128 tokens, 127 scored', rowIndices: S.data.ids.tiny }
    },
    results: {
      pretrained: { pubmedHeldOut: lm(R.pre_pubmed.items, 30001), medicalQaHeldOut: lm(R.pre_qa.items, 30001), medmcqa: mcq(preMcq), routing: routing(sumRoute(R.pre_pubmed.route, R.pre_qa.route)), decodeTokensPerSec: r(R.pre_speed.items[0].tokPerSec, 1) },
      sft: { pubmedHeldOut: lm(R.sft_pubmed.items, 30001), medicalQaHeldOut: lm(R.sft_qa.items, 30001), medmcqa: mcq(sftMcq), routing: routing(sumRoute(R.sft_pubmed.route, R.sft_qa.route)), decodeTokensPerSec: r(R.sft_speed.items[0].tokPerSec, 1) },
      tiny: { tinyStoriesValidation: lm(R.tiny_lm.items, 3512), decodeTokensPerSec: r(R.tiny_speed.items[0].tokPerSec, 1), modelCardValidationLoss: 6.3688 }
    },
    comparisons: {
      sftVsPretrainedQaPerplexityRatio: paired(R.pre_qa.items, R.sft_qa.items),
      sftVsPretrainedPubmedPerplexityRatio: paired(R.pre_pubmed.items, R.sft_pubmed.items),
      medmcqaBothCorrect: bothRight
    },
    samples: {
      prompts: S.genPrompts,
      pretrained: R.pre_gen.items.map(function (g) { return g.text; }),
      sft: R.sft_gen.items.map(function (g) { return g.text; }),
      tiny: { prompt: 'Once upon a time', text: R.tiny_gen.items[0].text },
      decoding: 'greedy (argmax), up to 60 new tokens, stops at <eos>'
    },
    seconds: Object.keys(R).reduce(function (o, k) { o[k] = Math.round(R[k].seconds); return o; }, {})
  };
  return out;
}
if (typeof module !== 'undefined') module.exports = aggregateEval;
