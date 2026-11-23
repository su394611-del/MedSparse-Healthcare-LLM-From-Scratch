"""Build /training-logs/tiny-gpt/index.html: Tiny GPT training log, designed to sit in an iframe.

Sources (all real):
  model card (README) of sudhanshu42844/my-tiny-llm: per-epoch train/val loss, 270/30 split, 2 epochs on a consumer-grade GPU
  config.json of the same repo: GPT-2 style, 3 layers, 4 heads, 128 hidden, 128 context, vocab 3,512
  transformer_hf_dataset_and_llm.py (run_generate_llm): logs one averaged line per epoch; defaults
  eval/results.json: browser re-measurement on TinyStories validation stories
"""
import html
import json
import math
import sys
from pathlib import Path

RES = json.load(open(sys.argv[1]))
OUT = Path(sys.argv[2])
TRAIN, VAL = '#8c78ee', '#1ca3b0'          # validated pair (dataviz validator, dark, surface #102137)
INK, MUTED, DIM, GRID, SURF = '#edf5ff', '#a8bdd4', '#87a3c1', '#274057', '#0f1f35'
EPOCHS = [(1, 7.5319, 7.0656), (2, 6.6421, 6.3688)]
UNI = math.log(3512)
tiny = RES['results']['tiny']['tinyStoriesValidation']
E = html.escape


def chart():
    W, H, L, R, T, B = 480, 300, 54, 132, 22, 50
    xs = lambda e: L + (W - L - R) * e / 2
    y0, y1 = 5.5, 8.5
    ys = lambda v: T + (H - T - B) * (y1 - v) / (y1 - y0)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="c-t c-d" xmlns="http://www.w3.org/2000/svg">',
         '<title id="c-t">Tiny GPT loss per epoch</title>',
         '<desc id="c-d">Training loss 7.53 then 6.64; validation loss 7.07 then 6.37, after epochs 1 and 2. '
         f'An untrained model starts near {UNI:.2f}. Browser re-check on 100 validation-split stories: {tiny["meanNll"]:.2f}.</desc>']
    for v in (5.5, 6, 6.5, 7, 7.5, 8, 8.5):
        y = ys(v)
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 10}" y="{y + 5:.1f}" fill="{DIM}" font-size="14" text-anchor="end">{v:g}</text>')
    for e in (0, 1, 2):
        o.append(f'<text x="{xs(e):.1f}" y="{H - B + 22}" fill="{DIM}" font-size="14" text-anchor="middle">{e}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 8}" fill="{MUTED}" font-size="13" text-anchor="middle">Epoch</text>')
    o.append(f'<text x="15" y="{(T + H - B) / 2:.1f}" fill="{MUTED}" font-size="13" text-anchor="middle" transform="rotate(-90 15 {(T + H - B) / 2:.1f})">Loss (nats/token)</text>')
    # untrained reference at epoch 0 (theoretical, not a logged value)
    o.append(f'<g><title>Untrained model: about {UNI:.2f} (uniform guess over 3,512 tokens; not logged)</title>'
             f'<circle cx="{xs(0):.1f}" cy="{ys(UNI):.1f}" r="5.5" fill="{SURF}" stroke="{DIM}" stroke-width="2"/></g>'
             f'<text x="{xs(0) + 12:.1f}" y="{ys(UNI) + 5:.1f}" fill="{MUTED}" font-size="13">Untrained ≈ {UNI:.2f}</text>')
    for name, color, idx in (('Training loss (epoch mean)', TRAIN, 1), ('Validation loss', VAL, 2)):
        pts = [(row[0], row[idx]) for row in EPOCHS]
        d = ' '.join(f'{"M" if i == 0 else "L"}{xs(e):.1f},{ys(v):.1f}' for i, (e, v) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        for e, v in pts:
            o.append(f'<g><title>{E(name)}, epoch {e}: {v:.4f} (perplexity {math.exp(v):,.0f})</title>'
                     f'<circle cx="{xs(e):.1f}" cy="{ys(v):.1f}" r="16" fill="transparent"/>'
                     f'<circle cx="{xs(e):.1f}" cy="{ys(v):.1f}" r="5.5" fill="{color}" stroke="{SURF}" stroke-width="2"/></g>')
    # browser re-check, drawn as a ring at epoch 2
    yb = ys(tiny['meanNll'])
    o.append(f'<g><title>Browser re-check: {tiny["meanNll"]:.4f} on {tiny["items"]} TinyStories validation-split stories (perplexity {tiny["perplexity"]:,.0f})</title>'
             f'<circle cx="{xs(2):.1f}" cy="{yb:.1f}" r="9" fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="3 3"/></g>')
    # direct labels, de-collided
    labels = sorted([(ys(6.6421), 'Train 6.64', TRAIN), (ys(6.3688), 'Val 6.37', VAL)])
    last = None
    for y, t, c in labels:
        if last is not None and y - last < 22:
            y = last + 22
        last = y
        o.append(f'<rect x="{xs(2) + 12:.1f}" y="{y - 6:.1f}" width="9" height="3" rx="1.5" fill="{c}"/>'
                 f'<text x="{xs(2) + 26:.1f}" y="{y + 1:.1f}" fill="{INK}" font-size="15" font-weight="700">{t}</text>')
    o.append(f'<text x="{xs(2) + 26:.1f}" y="{max(last + 24, yb + 24):.1f}" fill="{MUTED}" font-size="12.5">◌ Browser {tiny["meanNll"]:.2f}</text>')
    o.append('</svg>')
    return '\n'.join(o)


rows = ''.join(
    f'<tr><td>{e}</td><td><b>{tr:.4f}</b></td><td><b>{va:.4f}</b></td><td>{math.exp(va):,.0f}</td><td>Saved</td></tr>'
    for e, tr, va in EPOCHS)

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1" name="viewport"/>
<meta content="#0b1628" name="theme-color"/>
<meta content="Training log of Tiny GPT, a 1.06M-parameter GPT-2-style model trained from scratch on TinyStories: loss per epoch, settings and an independent re-check." name="description"/>
<link href="https://su394611-del.github.io/training-logs/tiny-gpt/" rel="canonical"/>
<link href="/assets/favicon.svg" rel="icon" type="image/svg+xml"/>
<title>Tiny GPT training log | Sudhanshu Shukla</title>
<style>
  *,*::before,*::after{{box-sizing:border-box}}
  html,body{{margin:0;background:#0b1628;color:{INK};font:14px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
  main{{max-width:1080px;margin:0 auto;padding:22px 22px 26px}}
  .kicker{{margin:0 0 4px;font-size:10.5px;font-weight:850;letter-spacing:.16em;text-transform:uppercase;color:#74e3ea}}
  h1{{margin:0 0 6px;font-size:21px;line-height:1.25}}
  .lede{{margin:0 0 16px;max-width:760px;color:{MUTED};font-size:13.5px}}
  .facts{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px;padding:0;list-style:none}}
  .facts li{{padding:5px 10px;border:1px solid #274057;border-radius:999px;font-size:12px;color:{MUTED};background:#102137}}
  .facts b{{color:{INK};font-weight:750}}
  .grid{{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:22px;align-items:start}}
  .panel{{padding:16px;border:1px solid #274057;border-radius:14px;background:linear-gradient(150deg,#142942,#102137)}}
  .legend{{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 6px;font-size:12.5px;color:{MUTED}}}
  .legend span{{display:inline-flex;align-items:center;gap:8px}}
  .legend i{{display:inline-block;width:18px;height:3px;border-radius:2px}}
  svg{{display:block;width:100%;height:auto;max-width:520px}}
  h2{{margin:0 0 8px;font-size:13px;font-weight:850;letter-spacing:.1em;text-transform:uppercase;color:{DIM}}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{padding:8px 6px;text-align:left;border-bottom:1px solid #274057;vertical-align:top}}
  th{{font-size:10.5px;font-weight:850;letter-spacing:.1em;text-transform:uppercase;color:{DIM}}}
  td{{color:{MUTED}}} td b{{color:{INK}}}
  .settings td:first-child{{color:{INK};font-weight:650;white-space:nowrap}}
  .tag{{display:inline-block;margin-left:6px;padding:0 6px;border-radius:6px;font-size:10.5px;font-weight:750;color:#0b1628;background:#87e1c8}}
  .tag.def{{background:#a8bdd4}}
  .note{{margin:12px 0 0;font-size:12.5px;color:{MUTED}}}
  .check{{margin:14px 0 0;padding:12px 14px;border:1px solid #274057;border-radius:10px;font-size:13px;color:{MUTED};background:#0e1c31}}
  .check b{{color:{INK}}}
  a{{color:#74e3ea}} a:focus-visible{{outline:3px solid #95eaff;outline-offset:2px;border-radius:4px}}
  .links{{display:flex;flex-wrap:wrap;gap:6px 16px;margin:16px 0 0;font-size:12.5px}}
  @media(max-width:820px){{.grid{{grid-template-columns:minmax(0,1fr)}}}}
  .panel{{min-width:0;overflow-x:auto}}
  @media(max-width:480px){{main{{padding:16px 14px 20px}}.panel{{padding:12px}}.settings td:first-child{{white-space:normal}}}}
</style>
</head>
<body>
<main>
  <p class="kicker">Training log · Tiny GPT</p>
  <h1>GPT-2-style model trained from random weights on TinyStories</h1>
  <p class="lede">Every value the training program logged for the published checkpoint. The program writes one averaged line per epoch, so this run has two logged points per curve.</p>
  <ul class="facts">
    <li><b>1,060,992</b> parameters</li><li><b>3</b> layers · <b>4</b> heads · <b>128</b> hidden</li><li><b>128</b>-token context</li>
    <li><b>3,512</b>-token BPE vocabulary</li><li><b>270</b> train / <b>30</b> validation stories</li><li><b>2</b> epochs on a consumer-grade GPU</li>
  </ul>
  <div class="grid">
    <section class="panel" aria-label="Loss per epoch">
      <div class="legend"><span><i style="background:{TRAIN}"></i>Training loss (mean over the epoch)</span><span><i style="background:{VAL}"></i>Validation loss (30 held-out stories)</span></div>
      {chart()}
      <p class="note">Training loss is averaged while the weights are still changing and with dropout on, so it sits above validation loss. The untrained point is the uniform-guess value, not a logged one.</p>
    </section>
    <section>
      <div class="panel">
        <h2>Logged values</h2>
        <table><thead><tr><th scope="col">Epoch</th><th scope="col">Train loss</th><th scope="col">Val loss</th><th scope="col">Val perplexity</th><th scope="col">Checkpoint</th></tr></thead>
        <tbody>{rows}</tbody></table>
        <p class="note">The program keeps the checkpoint with the lowest validation loss, so the published weights are from epoch 2.</p>
      </div>
      <div class="check"><b>Independent re-check.</b> The published weights, run in the browser on the first {tiny['items']} stories of TinyStories' own validation file, score a loss of <b>{tiny['meanNll']:.3f}</b> (perplexity {tiny['perplexity']:,.0f}, 95% interval {tiny['ci95'][0]:,.0f}–{tiny['ci95'][1]:,.0f}), within 0.4% of the logged {EPOCHS[-1][2]:.4f}.</div>
    </section>
  </div>
  <section class="panel" style="margin-top:18px" aria-label="Run settings">
    <h2>Run settings</h2>
    <table class="settings"><tbody>
      <tr><td>Program</td><td><code>transformer_hf_dataset_and_llm.py --generate-llm</code> (run_generate_llm)<span class="tag">recorded</span></td></tr>
      <tr><td>Data</td><td>roneneldan/TinyStories, train split; 270 training and 30 validation stories (10% held out)<span class="tag">recorded</span></td></tr>
      <tr><td>Tokenizer</td><td>Byte-level BPE fitted on the training stories only; 3,512 tokens learned<span class="tag">recorded</span></td></tr>
      <tr><td>Model</td><td>GPT2LMHeadModel from random weights: 3 layers, 4 heads, 128 hidden, 128 positions, tied embeddings<span class="tag">recorded</span></td></tr>
      <tr><td>Optimizer</td><td>AdamW, weight decay 0.01, gradient clipping 1.0<span class="tag">recorded</span></td></tr>
      <tr><td>Batch size · LR</td><td>8 · 2e-4 (34 optimizer steps per epoch)<span class="tag def">script defaults</span></td></tr>
    </tbody></table>
    <p class="note">“Recorded” values come from the model card, config.json and the training code; “script defaults” are the program's defaults, consistent with the logged 270/30 split and 2 epochs, because the exact command line for this run was not saved.</p>
  </section>
  <p class="links"><a href="https://huggingface.co/sudhanshu42844/my-tiny-llm" target="_blank" rel="noopener noreferrer">Model card ↗</a><a href="/playground/?model=tiny" target="_top">Run it in the browser</a><a href="/#model-evaluation" target="_top">Model evaluation</a></p>
</main>
<script>
  // When embedded, tell the parent page how tall this content is (same origin).
  (function () {{
    if (window.self === window.top) return;
    function send() {{ try {{ parent.postMessage({{ type: 'tiny-gpt-log-height', height: document.documentElement.scrollHeight }}, location.origin); }} catch (e) {{}} }}
    window.addEventListener('load', send);
    window.addEventListener('resize', send);
    if (window.ResizeObserver) new ResizeObserver(send).observe(document.body);
  }})();
</script>
<script src="/assets/privacy.js" defer></script>
</body>
</html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page, encoding='utf8')
print('wrote', OUT, len(page), 'bytes')
