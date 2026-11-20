/*
 * Evaluation worker for the Model Playground engine.
 * Runs after the playground's engine source (self.PlaygroundEngine), with one line of the
 * engine patched to report which experts the MoE router picks (self.__route).
 * Everything here is deterministic: no sampling, greedy decoding only.
 */
'use strict';
var EV = (function () {
  var E = (typeof self !== 'undefined' && self.PlaygroundEngine) || (typeof global !== 'undefined' && global.PlaygroundEngine);
  var CACHE_NAME = 'ss-model-playground-v1';
  var SFT_SYSTEM = 'Provide accurate educational medical information, and acknowledge uncertainty.';
  var SEQ = 128;                      // training sequence length for all three models

  var route = null;                   // route[layer][expert] = number of top-2 selections
  var root = typeof self !== 'undefined' ? self : global;
  root.__route = function (l, chosen) { if (route) for (var i = 0; i < chosen.length; i++) route[l][chosen[i]]++; };

  function fetchBuf(url) {
    var tryCache = (typeof caches !== 'undefined') ? caches.open(CACHE_NAME).then(function (c) { return c.match(url); }).catch(function () { return null; }) : Promise.resolve(null);
    return tryCache.then(function (hit) {
      if (hit) return hit.arrayBuffer();
      return fetch(url).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url); return r.arrayBuffer(); });
    });
  }

  function load(repo, sha, kind) {
    var base = 'https://huggingface.co/' + repo + '/resolve/' + sha + '/';
    return Promise.all([fetchBuf(base + 'config.json'), fetchBuf(base + 'tokenizer.json'), fetchBuf(base + 'model.safetensors')]).then(function (b) {
      var dec = new TextDecoder();
      var cfg = JSON.parse(dec.decode(b[0]));
      var tok = new E.ByteLevelBPE(JSON.parse(dec.decode(b[1])));
      var model = kind === 'mixtral' ? new E.MixtralModel(cfg, E.parseSafetensors(b[2])) : new E.GPT2Model(cfg, E.parseSafetensors(b[2]));
      if (tok.size !== model.V) throw new Error('vocab mismatch');
      return { cfg: cfg, tok: tok, model: model, kind: kind };
    });
  }

  function logSoftmaxAt(logits, t) {
    var mx = -Infinity, i, s = 0;
    for (i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    for (i = 0; i < logits.length; i++) s += Math.exp(logits[i] - mx);
    return logits[t] - mx - Math.log(s);
  }
  function argmax(logits) {
    var b = 0;
    for (var i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i;
    return b;
  }

  // Sum of -log p(ids[i] | ids[<i]) for i >= scoreFrom; also next-token top-1 hits.
  function scoreSeq(m, ids, scoreFrom) {
    m.model.reset();
    var nll = 0, n = 0, hit = 0;
    for (var i = 0; i < ids.length - 1; i++) {
      var want = i + 1 >= scoreFrom;
      var logits = m.model.step(ids[i], want);
      if (want) {
        nll -= logSoftmaxAt(logits, ids[i + 1]);
        if (argmax(logits) === ids[i + 1]) hit++;
        n++;
      }
    }
    return { nll: nll, n: n, top1: hit };
  }

  // PubMed held-out: packed-pretraining format (document tokens + <eos>, no <bos>), first 129 tokens.
  function pubmedIds(m, text) {
    var eos = m.tok.tokenToId('<eos>');
    var ids = m.tok.encode(text.slice(0, 4000));
    if (ids.length < SEQ + 1) ids = ids.concat([eos]);
    return ids.slice(0, SEQ + 1);
  }

  // Mirrors medical_data.tokenize_example(seq_len=128): completion-only targets, prompt kept from the right.
  function sftExample(m, question, answer) {
    var bos = m.tok.tokenToId('<bos>'), eos = m.tok.tokenToId('<eos>');
    var header = '### System:\n' + SFT_SYSTEM + '\n\n### User:\n' + question + '\n\n### Assistant:\n';
    var prompt = (bos != null ? [bos] : []).concat(m.tok.encode(header));
    var ans = m.tok.encode(answer).concat([eos]);
    var budget = SEQ + 1;
    if (ans.length > budget - 2) { ans = ans.slice(0, budget - 2); ans[ans.length - 1] = eos; }
    var remaining = budget - ans.length;
    var questionCut = prompt.length > remaining;
    prompt = prompt.slice(-remaining);
    return { ids: prompt.concat(ans), scoreFrom: prompt.length, questionCut: questionCut };
  }

  // MedMCQA cloze scoring: log p(option | SFT-formatted question), prompt KV cache reused per option.
  function mcqItem(m, item) {
    var bos = m.tok.tokenToId('<bos>');
    var header = '### System:\n' + SFT_SYSTEM + '\n\n### User:\n' + item.q + '\n\n### Assistant:\n';
    var opts = item.opts.map(function (o) { return m.tok.encode(o); });
    var longest = Math.max.apply(null, opts.map(function (o) { return o.length; }));
    var prompt = (bos != null ? [bos] : []).concat(m.tok.encode(header));
    prompt = prompt.slice(-Math.max(8, SEQ - longest));
    m.model.reset();
    var last = null;
    for (var i = 0; i < prompt.length; i++) last = m.model.step(prompt[i], i === prompt.length - 1);
    last = Float32Array.from(last);
    var P = m.model.pos, sums = [], means = [];
    for (var k = 0; k < opts.length; k++) {
      var o = opts[k].slice(0, SEQ - P);
      m.model.pos = P;                                   // rewind: later K/V slots are overwritten
      var lp = logSoftmaxAt(last, o[0]);
      for (var j = 0; j < o.length - 1; j++) lp += logSoftmaxAt(m.model.step(o[j], true), o[j + 1]);
      sums.push(lp); means.push(lp / o.length);
    }
    var pick = function (a) { var b = 0; for (var x = 1; x < a.length; x++) if (a[x] > a[b]) b = x; return b; };
    return { predNorm: pick(means), predSum: pick(sums), cop: item.cop, promptTokens: prompt.length };
  }

  function greedy(m, text, maxNew, sft) {
    var bos = m.kind === 'mixtral' ? m.tok.tokenToId('<bos>') : null, eos = m.tok.tokenToId('<eos>');
    if (sft) text = '### System:\n' + SFT_SYSTEM + '\n\n### User:\n' + text + '\n\n### Assistant:\n';
    var ids = (bos != null ? [bos] : []).concat(m.tok.encode(text));
    ids = ids.slice(-(Math.min(SEQ, m.model.maxLen) - maxNew));      // prompt + new tokens stay within 128
    m.model.reset();
    var logits = null, out = [], t0 = Date.now();
    for (var i = 0; i < ids.length; i++) logits = m.model.step(ids[i], i === ids.length - 1);
    for (var s = 0; s < maxNew; s++) {
      var nxt = argmax(logits);
      if (nxt === eos) break;
      out.push(nxt);
      if (m.model.pos >= m.model.maxLen) break;
      logits = m.model.step(nxt, true);
    }
    return { text: m.tok.decode(out), tokens: out.length, ms: Date.now() - t0, promptTokens: ids.length };
  }

  function speed(m, n) {
    var bos = m.kind === 'mixtral' ? m.tok.tokenToId('<bos>') : null;
    var ids = (bos != null ? [bos] : []).concat(m.tok.encode('The patient was'));
    m.model.reset();
    var logits = null;
    for (var i = 0; i < ids.length; i++) logits = m.model.step(ids[i], true);
    var t0 = performance.now();
    for (var s = 0; s < n; s++) logits = m.model.step(argmax(logits), true);
    return n / ((performance.now() - t0) / 1000);
  }

  function newRoute(m) {
    if (m.kind !== 'mixtral') return null;
    var r = [];
    for (var l = 0; l < m.model.layers.length; l++) { r.push([]); for (var e = 0; e < m.model.E; e++) r[l].push(0); }
    return r;
  }

  // job: {repo, sha, kind, task: 'pubmed'|'qa'|'tiny'|'mcq'|'gen'|'speed', items, ...}
  function run(job, progress) {
    return load(job.repo, job.sha, job.kind).then(function (m) {
      var res = { task: job.task, repo: job.repo, sha: job.sha, items: [] };
      var t0 = Date.now();
      // A plain loop: the worker owns its thread, and timers in a hidden page's workers can be
      // throttled to about one per minute, so the loop never yields to setTimeout.
      route = (job.task === 'pubmed' || job.task === 'qa') ? newRoute(m) : null;
      for (var i = 0; i < job.items.length; i++) {
        var it = job.items[i], r;
        if (job.task === 'pubmed') r = scoreSeq(m, pubmedIds(m, it.text), 1);
        else if (job.task === 'tiny') r = scoreSeq(m, m.tok.encode(it.text).slice(0, SEQ), 1);
        else if (job.task === 'qa') { var ex = sftExample(m, it.q, it.a); r = scoreSeq(m, ex.ids, ex.scoreFrom); r.questionCut = ex.questionCut; }
        else if (job.task === 'mcq') r = mcqItem(m, it);
        else if (job.task === 'gen') { r = greedy(m, it.text, job.maxNew || 60, !!it.sft); r.id = it.id; }
        else if (job.task === 'speed') r = { tokPerSec: speed(m, it.n || 64) };
        res.items.push(r);
        if (progress) progress(i + 1, job.items.length);
      }
      res.route = route; res.seconds = (Date.now() - t0) / 1000; route = null;
      return res;
    });
  }

  if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof window === 'undefined') {
    self.onmessage = function (ev) {
      run(ev.data, function (done, total) { self.postMessage({ type: 'progress', done: done, total: total }); })
        .then(function (res) { self.postMessage({ type: 'done', res: res }); })
        .catch(function (e) { self.postMessage({ type: 'error', message: String(e && e.stack || e) }); });
    };
  }
  return { run: run, sftExample: sftExample, pubmedIds: pubmedIds, mcqItem: mcqItem, scoreSeq: scoreSeq, greedy: greedy, logSoftmaxAt: logSoftmaxAt, setRoute: function (r) { route = r; }, getRoute: function () { return route; }, newRoute: newRoute };
})();
if (typeof module !== 'undefined') module.exports = EV;
