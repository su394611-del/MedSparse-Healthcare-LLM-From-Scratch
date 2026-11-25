/*
 * Aggregates the Release 03 (MIRIAD SFT) run with the same formulas as aggregate.js.
 * Perplexity CIs: 2,000-sample bootstrap over evaluation items (documents / questions / stories),
 * token-weighted, fixed seed. Accuracy CIs: Wilson score interval. Paired SFT-vs-Pretrained
 * difference: paired bootstrap over the same items.
 */
function aggregateMiriad(S) {
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
  var mcqItems = R.miriad_mcq_a.items.concat(R.miriad_mcq_b.items);
  return {
    evaluated: S.finished, started: S.started,
    environment: { browser: (navigator.userAgentData && navigator.userAgentData.brands || []).map(function (b) { return b.brand + ' ' + b.version; }).join(', ') || navigator.userAgent, cpuThreads: navigator.hardwareConcurrency, engine: 'Model Playground JavaScript engine (float32, CPU, Web Worker)' },
    models: { sft: { repo: 'sudhanshu42844/medsparse-healthcare-medical-sft', revision: S.shas.sft }, miriad: { repo: 'sudhanshu42844/medsparse-healthcare-miriad-sft', revision: S.shas.miriad, parameters: 38779008 } },
    data: { counts: S.data.counts, rowIndices: S.data.ids, pubmedInfo: S.data.pubInfo || null, qaInfo: S.data.qaInfo || null },
    results: {
      miriad: { pubmedHeldOut: lm(R.miriad_pubmed.items, 30001), medicalQaHeldOut: lm(R.miriad_qa.items, 30001), medmcqa: mcq(mcqItems), routing: routing(sumRoute(R.miriad_pubmed.route, R.miriad_qa.route)), decodeTokensPerSec: r(R.miriad_speed.items[0].tokPerSec, 1) },
      sftReproduction: { pubmedHeldOut: lm(R.sft_pubmed.items, 30001), medicalQaHeldOut: lm(R.sft_qa.items, 30001) }
    },
    paired: { miriadVsSftQaPerplexityRatio: paired(R.sft_qa.items, R.miriad_qa.items), miriadVsSftPubmedPerplexityRatio: paired(R.sft_pubmed.items, R.miriad_pubmed.items) },
    samples: { prompts: S.genPrompts, miriad: R.miriad_gen.items.map(function (g) { return g.text; }) },
    seconds: Object.keys(R).reduce(function (o, k) { o[k] = Math.round(R[k].seconds); return o; }, {})
  };
}
