/*
 * Evaluation driver (runs in the playground page). Selects held-out data, runs workers, aggregates.
 * Needs: Split (split.js) and WORKER_SRC (worker.js text) defined beforehand.
 */
(function () {
  'use strict';
  var N = { pubmed: 40, qa: 80, mcq: 150, tiny: 100 };
  var REPOS = {
    pre: { repo: 'sudhanshu42844/medsparse-healthcare-pretrained-from-scratch', kind: 'mixtral' },
    sft: { repo: 'sudhanshu42844/medsparse-healthcare-medical-sft', kind: 'mixtral' },
    tiny: { repo: 'sudhanshu42844/my-tiny-llm', kind: 'gpt2' }
  };
  var S = window.__EVAL = { status: 'starting', log: [], progress: {}, data: {}, raw: {}, started: new Date().toISOString() };
  function log(m) { S.log.push(new Date().toISOString().slice(11, 19) + ' ' + m); }

  // The datasets server drops some large responses or rate-limits; back off and retry (2s, 4s, ... 30s).
  function getJSON(url, attempt) {
    attempt = attempt || 0;
    return fetch(url).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .catch(function (e) {
        if (attempt >= 9) throw e;
        S.retries = (S.retries || 0) + 1;
        return new Promise(function (res) { setTimeout(res, Math.min(30000, 2000 * Math.pow(2, attempt))); })
          .then(function () { return getJSON(url, attempt + 1); });
      });
  }
  function rows(ds, cfg, split, offset, length) {
    return getJSON('https://datasets-server.huggingface.co/rows?dataset=' + encodeURIComponent(ds) + '&config=' + cfg + '&split=' + split + '&offset=' + offset + '&length=' + length)
      .then(function (j) { return j.rows.map(function (r) { return { idx: r.row_idx, row: r.row }; }); });
  }

  // healthcare_corpus.license_ok / medical_filter (JS ports)
  var TERMS = ['patient', 'patients', 'clinical', 'medical', 'healthcare', 'hospital', 'physician', 'medicine', 'disease', 'diagnos', 'treatment', 'therap', 'symptom', 'syndrome', 'infection', 'infectious', 'cancer', 'tumor', 'tumour', 'oncolog', 'diabet', 'cardiac', 'cardiolog', 'heart', 'stroke', 'neurolog', 'brain', 'surgery', 'blood', 'respiratory', 'pulmonary', 'pharmacolog', 'drug', 'vaccine', 'immunolog', 'patholog', 'epidemiolog', 'pregnan', 'pediatr', 'renal', 'kidney', 'liver', 'fracture', 'fever', 'antibiotic', 'mortality'];
  function medicalFilter(t) { var s = t.toLowerCase(), c = 0; for (var i = 0; i < TERMS.length && c < 1; i++) c += s.split(TERMS[i]).length - 1; return c >= 1; }
  function licenseOk(v) {
    if (typeof v !== 'string') return false;
    var s = v.trim().toLowerCase().replace(/_/g, '-');
    if (['noncommercial', 'non-commercial', 'by-nc', 'sharealike', 'share-alike', 'by-sa', 'all rights reserved', 'by-nd'].some(function (t) { return s.indexOf(t) !== -1; })) return false;
    if (['cc0', 'cc0-1.0', 'public domain', 'cc-by-4.0', 'cc-by-3.0', 'cc-by-2.0'].indexOf(s) !== -1) return true;
    if (s.indexOf('creativecommons.org/publicdomain/zero/1.0') !== -1) return true;
    return /creativecommons\.org\/licenses\/by\/(?:2\.0|3\.0|4\.0)(?:\/|$)/.test(s);
  }

  // Streams the first original shard (JSON lines, gzip) and keeps the first N.pubmed documents, in file
  // order, that pass the training filters and fall in the 2% validation partition. Stops reading after that.
  var PUBMED_URL = 'https://huggingface.co/datasets/common-pile/pubmed_filtered/resolve/main/licensed_pubmed-0000.json.gz';
  function selectPubmed() {
    var picked = [], scanned = 0, rejected = { license: 0, clinical: 0, train: 0 }, ctrl = new AbortController(), buf = '';
    function consider(row, idx) {
      var t = row.text, md = row.metadata || {};
      if (typeof t !== 'string' || !t.trim()) return;
      if (!medicalFilter(t)) { rejected.clinical++; return; }
      if (!licenseOk(md.license || row.license)) { rejected.license++; return; }
      if (Split.pretrainPartition(t) !== 'validation') { rejected.train++; return; }
      picked.push({ idx: idx, id: row.id, text: t.slice(0, 4000) });
    }
    return fetch(PUBMED_URL, { signal: ctrl.signal }).then(function (res) {
      if (!res.ok) throw new Error('PubMed shard HTTP ' + res.status);
      var reader = res.body.pipeThrough(new DecompressionStream('gzip')).pipeThrough(new TextDecoderStream()).getReader();
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return;
          buf += r.value;
          var nl;
          while (picked.length < N.pubmed && (nl = buf.indexOf('\n')) !== -1) {
            var line = buf.slice(0, nl); buf = buf.slice(nl + 1);
            if (!line.trim()) continue;
            consider(JSON.parse(line), scanned++);
          }
          S.progress.select_pubmed = picked.length + '/' + N.pubmed + ' (scanned ' + scanned + ')';
          if (picked.length >= N.pubmed) { try { ctrl.abort(); } catch (e) { /* done */ } return; }
          return pump();
        });
      }
      return pump();
    }).then(function () {
      S.data.pubmedInfo = { file: 'licensed_pubmed-0000.json.gz', scanned: scanned, rejected: rejected, selected: picked.length };
      return picked;
    });
  }
  // medical-o1 'en' config = medical_o1_sft.json; rows in file order, 5% validation bucket on the question.
  function selectQA() {
    return fetch('https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/resolve/main/medical_o1_sft.json')
      .then(function (r) { if (!r.ok) throw new Error('medical-o1 HTTP ' + r.status); return r.json(); })
      .then(function (all) {
        var picked = [], scanned = 0;
        for (var i = 0; i < all.length && picked.length < N.qa; i++) {
          scanned++;
          var q = (all[i].Question || '').trim(), a = (all[i].Response || '').trim();
          if (!q || !a) continue;
          if (Split.sftBucket(q) === 'validation') picked.push({ idx: i, q: q, a: a });
        }
        S.progress.select_qa = picked.length + '/' + N.qa + ' (scanned ' + scanned + ')';
        S.data.qaInfo = { file: 'medical_o1_sft.json', totalRows: all.length, scanned: scanned, selected: picked.length };
        return picked;
      });
  }
  function selectMCQ() {
    return Promise.all([rows('openlifescienceai/medmcqa', 'default', 'validation', 0, 100), rows('openlifescienceai/medmcqa', 'default', 'validation', 100, 100)])
      .then(function (pages) {
        var out = [];
        pages[0].concat(pages[1]).forEach(function (x) {
          if (out.length >= N.mcq) return;
          var r = x.row, opts = [r.opa, r.opb, r.opc, r.opd].map(function (o) { return (o || '').trim(); });
          if (!r.question || opts.some(function (o) { return !o; }) || !(r.cop >= 0 && r.cop <= 3)) return;
          out.push({ idx: x.idx, q: r.question.trim(), opts: opts, cop: r.cop, subject: r.subject_name });
        });
        S.data.mcqInfo = { selected: out.length, firstCop: pages[0][0].row.cop, firstOpa: pages[0][0].row.opa };
        return out;
      });
  }
  // TinyStories validation text file: stories are separated by <|endoftext|>; first N.tiny stories.
  function selectTiny() {
    return fetch('https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt', { headers: { Range: 'bytes=0-300000' } })
      .then(function (r) { if (!r.ok) throw new Error('TinyStories HTTP ' + r.status); return r.text(); })
      .then(function (txt) {
        var stories = txt.split('<|endoftext|>').map(function (x) { return x.trim(); }).filter(Boolean);
        stories.pop();   // the last piece may be cut by the byte range
        return stories.slice(0, N.tiny).map(function (t, i) { return { idx: i, text: t }; });
      });
  }

  function hfSha(repo) { return getJSON('https://huggingface.co/api/models/' + repo).then(function (j) { return j.sha; }); }

  var queue = [], active = 0, MAX = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  var engineSource;
  function runJob(name, job) {
    return new Promise(function (resolve, reject) {
      queue.push(function () {
        active++;
        var src = engineSource;
        var line = 'const chosen = order.slice(0, this.topK);';
        if (src.indexOf(line) === -1) { reject(new Error('engine changed: router hook not found')); return; }
        src = src.replace(line, line + ' if (self.__route) self.__route(l, chosen);');
        var w = new Worker(URL.createObjectURL(new Blob([src + '\n' + window.__EV_WORKER_SRC], { type: 'text/javascript' })));
        S.progress[name] = 'loading';
        w.onmessage = function (ev) {
          var d = ev.data;
          if (d.type === 'progress') S.progress[name] = d.done + '/' + d.total;
          else if (d.type === 'done') { S.progress[name] = 'done ' + Math.round(d.res.seconds) + 's'; S.raw[name] = d.res; w.terminate(); active--; pump(); resolve(d.res); }
          else if (d.type === 'error') { S.progress[name] = 'ERROR'; log(name + ': ' + d.message); w.terminate(); active--; pump(); reject(new Error(d.message)); }
        };
        w.onerror = function (e) { S.progress[name] = 'ERROR'; log(name + ' worker error ' + e.message); active--; pump(); reject(e); };
        w.postMessage(job);
      });
      pump();
    });
  }
  function pump() { while (active < MAX && queue.length) queue.shift()(); }

  S.run = Promise.all([hfSha(REPOS.pre.repo), hfSha(REPOS.sft.repo), hfSha(REPOS.tiny.repo), fetch(new URL('engine-src.js', document.baseURI)).then(function (response) {
    if (!response.ok) throw new Error('Cannot load playground engine: ' + response.status);
    return response.text();
  })]).then(function (shas) {
    engineSource = shas[3];
    REPOS.pre.sha = shas[0]; REPOS.sft.sha = shas[1]; REPOS.tiny.sha = shas[2];
    S.shas = { pre: shas[0], sft: shas[1], tiny: shas[2] };
    S.status = 'selecting data';
    return Promise.all([selectPubmed(), selectQA(), selectMCQ(), selectTiny()]);
  }).then(function (d) {
    var pub = d[0], qa = d[1], mcq = d[2], tiny = d[3];
    S.data.counts = { pubmed: pub.length, qa: qa.length, mcq: mcq.length, tiny: tiny.length };
    S.data.ids = { pubmed: pub.map(function (x) { return x.idx; }), qa: qa.map(function (x) { return x.idx; }), mcq: mcq.map(function (x) { return x.idx; }), tiny: tiny.map(function (x) { return x.idx; }) };
    S.data.mcqSubjects = mcq.map(function (x) { return x.subject; });
    S.status = 'evaluating';
    log('data ready ' + JSON.stringify(S.data.counts));
    var half = Math.ceil(mcq.length / 2);
    var J = function (k, task, items, extra) { var j = { repo: REPOS[k].repo, sha: REPOS[k].sha, kind: REPOS[k].kind, task: task, items: items }; for (var e in extra || {}) j[e] = extra[e]; return j; };
    var genQs = qa.slice(0, 3).map(function (x, i) { return { id: 'qa' + i, text: x.q, sft: true }; });
    return Promise.all([
      runJob('pre_pubmed', J('pre', 'pubmed', pub)), runJob('sft_pubmed', J('sft', 'pubmed', pub)),
      runJob('pre_qa', J('pre', 'qa', qa)), runJob('sft_qa', J('sft', 'qa', qa)),
      runJob('pre_mcq_a', J('pre', 'mcq', mcq.slice(0, half))), runJob('pre_mcq_b', J('pre', 'mcq', mcq.slice(half))),
      runJob('sft_mcq_a', J('sft', 'mcq', mcq.slice(0, half))), runJob('sft_mcq_b', J('sft', 'mcq', mcq.slice(half))),
      runJob('tiny_lm', J('tiny', 'tiny', tiny)),
      runJob('pre_gen', J('pre', 'gen', genQs, { maxNew: 60 })), runJob('sft_gen', J('sft', 'gen', genQs, { maxNew: 60 })),
      runJob('tiny_gen', J('tiny', 'gen', [{ id: 'story', text: 'Once upon a time', sft: false }], { maxNew: 60 }))
    ]).then(function () {
      S.status = 'speed test';
      MAX = 1;   // speed is measured with nothing else running
      return Promise.all([runJob('pre_speed', J('pre', 'speed', [{ n: 64 }])), runJob('sft_speed', J('sft', 'speed', [{ n: 64 }])), runJob('tiny_speed', J('tiny', 'speed', [{ n: 64 }]))]);
    }).then(function () {
      S.genPrompts = genQs.map(function (g) { return g.text; });
      S.status = 'done';
      S.finished = new Date().toISOString();
      log('all done');
    });
  }).catch(function (e) { S.status = 'failed'; log('FAILED ' + (e && e.stack || e)); });
})();
