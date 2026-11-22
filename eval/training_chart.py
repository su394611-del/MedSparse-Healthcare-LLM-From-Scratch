"""Static SVG training charts for the MedSparse Healthcare project page (no JavaScript needed).

stage_chart(results)  -> held-out loss at each published checkpoint (always available).
step_chart(dir)       -> per-step training curve, only when real metrics files exist:
                         eval/training/pretrain_metrics.jsonl and/or eval/training/sft_metrics.jsonl
                         (the metrics.jsonl files written by MetricsLog during training).
Palette checked with the dataviz validator on the card surface #102137 (dark): PASS.
"""
import html
import json
import math
from pathlib import Path

PUBMED, QA = '#1ca3b0', '#8c78ee'      # categorical, fixed order
INK, MUTED, DIM, GRID, SURFACE = '#edf5ff', '#a8bdd4', '#87a3c1', '#274057', '#102137'
PRETRAIN_STEPS, SFT_STEPS = 5000, 2000
E = html.escape


def _ln(x):
    return math.log(x)


def _fmt(x):
    return f'{x:.2f}'


def stage_chart(res):
    """Loss (nats/token) at step 0 (uniform guess), 5,000 (pretrained) and 7,000 (after SFT)."""
    R = res['results']
    W, H = 480, 320
    L, Rm, T, B = 54, 126, 30, 50
    xs = lambda s: L + (W - L - Rm) * s / (PRETRAIN_STEPS + SFT_STEPS)
    y0, y1 = 5.0, 11.0
    ys = lambda v: T + (H - T - B) * (y1 - v) / (y1 - y0)
    uni = _ln(30001)
    series = [
        ('PubMed held-out', PUBMED, 'pubmedHeldOut', f"{res['data']['pubmed']['selected']} articles"),
        ('Medical Q&A held-out', QA, 'medicalQaHeldOut', f"{res['data']['qa']['selected']} answers"),
    ]
    o = [f'<svg class="me-chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="tc-title tc-desc" xmlns="http://www.w3.org/2000/svg">',
         '<title id="tc-title">Held-out loss at each training stage</title>',
         f'<desc id="tc-desc">Cross-entropy on held-out data. Untrained guess {_fmt(uni)} at step 0; after 5,000 pretraining steps PubMed '
         f'{_fmt(R["pretrained"]["pubmedHeldOut"]["meanNll"])} and Q&amp;A {_fmt(R["pretrained"]["medicalQaHeldOut"]["meanNll"])}; '
         f'after 2,000 more SFT steps PubMed {_fmt(R["sft"]["pubmedHeldOut"]["meanNll"])} and Q&amp;A {_fmt(R["sft"]["medicalQaHeldOut"]["meanNll"])}.</desc>']
    # stage bands
    o.append(f'<rect x="{xs(0):.1f}" y="{T}" width="{xs(PRETRAIN_STEPS) - xs(0):.1f}" height="{H - T - B}" fill="#14304a" opacity=".35"/>')
    o.append(f'<rect x="{xs(PRETRAIN_STEPS):.1f}" y="{T}" width="{xs(7000) - xs(PRETRAIN_STEPS):.1f}" height="{H - T - B}" fill="#2a2257" opacity=".35"/>')
    o.append(f'<text x="{(xs(0) + xs(PRETRAIN_STEPS)) / 2:.1f}" y="{T - 10}" fill="{DIM}" font-size="12.5" font-weight="700" text-anchor="middle" letter-spacing=".08em">PRETRAINING</text>')
    o.append(f'<text x="{(xs(PRETRAIN_STEPS) + xs(7000)) / 2:.1f}" y="{T - 10}" fill="{DIM}" font-size="12.5" font-weight="700" text-anchor="middle" letter-spacing=".08em">SFT</text>')
    # grid + y axis
    for v in range(5, 12):
        y = ys(v)
        o.append(f'<line x1="{L}" x2="{W - Rm}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{L - 10}" y="{y + 5:.1f}" fill="{DIM}" font-size="14" text-anchor="end">{v}</text>')
    o.append(f'<text x="15" y="{(T + H - B) / 2:.1f}" fill="{MUTED}" font-size="13" text-anchor="middle" transform="rotate(-90 15 {(T + H - B) / 2:.1f})">Loss (nats/token), lower is better</text>')
    for s in range(0, 7001, 1000):
        x = xs(s)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{H - B}" y2="{H - B + 5}" stroke="{GRID}"/>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 20}" fill="{DIM}" font-size="14" text-anchor="middle">{s // 1000}k</text>' if s else
                 f'<text x="{x:.1f}" y="{H - B + 20}" fill="{DIM}" font-size="14" text-anchor="middle">0</text>')
    o.append(f'<text x="{(L + W - Rm) / 2:.1f}" y="{H - 8}" fill="{MUTED}" font-size="13" text-anchor="middle">Optimizer steps</text>')
    # series
    stage_names = {0: 'untrained (uniform guess over 30,001 tokens)', 5000: 'after 5,000 pretraining steps', 7000: 'after 2,000 SFT steps'}
    marks, labels = [], []
    for name, color, key, size in series:
        pts = [(0, uni, None, None), (5000, R['pretrained'][key]['meanNll'], R['pretrained'][key]['perplexity'], R['pretrained'][key]['ci95']),
               (7000, R['sft'][key]['meanNll'], R['sft'][key]['perplexity'], R['sft'][key]['ci95'])]
        d = ' '.join(f'{"M" if i == 0 else "L"}{xs(s):.1f},{ys(v):.1f}' for i, (s, v, _, _) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 5" stroke-linecap="round"/>')
        for s, v, p, ci in pts[1:]:
            lo, hi = _ln(ci[0]), _ln(ci[1])
            x = xs(s)
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ys(hi):.1f}" y2="{ys(lo):.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round" opacity=".55"/>')
        for s, v, p, ci in pts:
            x, y = xs(s), ys(v)
            tip = (f'{name} · {stage_names[s]}: loss {_fmt(v)} nats/token' +
                   (f' (perplexity {p:,.0f}; 95% interval {ci[0]:,.0f}–{ci[1]:,.0f})' if p else ' (perplexity 30,001)'))
            marks.append(f'<g><title>{E(tip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="transparent"/>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/></g>')
        last = pts[-1]
        labels.append((ys(last[1]), f'{name.split(" ")[0] if name.startswith("PubMed") else "Q&amp;A"} {_fmt(last[1])}', color))
    o.extend(marks)
    # direct labels at the right end, de-collided
    labels.sort()
    placed = []
    for y, text, color in labels:
        if placed and y - placed[-1] < 20:
            y = placed[-1] + 20
        placed.append(y)
        o.append(f'<rect x="{xs(7000) + 9:.1f}" y="{y - 6:.1f}" width="9" height="3" rx="1.5" fill="{color}"/>')
        o.append(f'<text x="{xs(7000) + 22:.1f}" y="{y + 1:.1f}" fill="{INK}" font-size="15" font-weight="700">{text}</text>')
    o.append(f'<text x="{xs(0) + 12:.1f}" y="{ys(uni) - 10:.1f}" fill="{MUTED}" font-size="13">Untrained: {_fmt(uni)}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def _read(path):
    rows = []
    for line in Path(path).read_text(encoding='utf8').splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def step_chart(training_dir):
    """Per-step curve from real MetricsLog files; returns None when no files are present."""
    d = Path(training_dir)
    pre_f, sft_f = d / 'pretrain_metrics.jsonl', d / 'sft_metrics.jsonl'
    if not pre_f.exists() and not sft_f.exists():
        return None
    pre = _read(pre_f) if pre_f.exists() else []
    sft = _read(sft_f) if sft_f.exists() else []
    tr_pre = [(r['step'], r['train_ce']) for r in pre if r.get('stage') == 'pretrain' and 'train_ce' in r]
    va_pre = [(r['step'], r['validation_ce']) for r in pre if r.get('stage') == 'pretrain-validation' and r.get('validation_ce') is not None]
    off = max([s for s, _ in tr_pre] or [PRETRAIN_STEPS])
    tr_sft = [(off + r['step'], r['train_completion_ce']) for r in sft if 'train_completion_ce' in r]
    va_sft = [(off + r['step'], r['val_completion_ce']) for r in sft if 'val_completion_ce' in r]
    allv = [v for _, v in tr_pre + va_pre + tr_sft + va_sft]
    if not allv:
        return None
    xmax = max(s for s, _ in tr_pre + va_pre + tr_sft + va_sft)
    W, H, L, Rm, T, B = 600, 320, 58, 24, 24, 52
    y0, y1 = math.floor(min(allv)), math.ceil(max(allv))
    xs = lambda s: L + (W - L - Rm) * s / xmax
    ys = lambda v: T + (H - T - B) * (y1 - v) / max(1e-9, (y1 - y0))
    o = [f'<svg class="me-chart" viewBox="0 0 {W} {H}" role="img" aria-label="Per-step training and validation loss" xmlns="http://www.w3.org/2000/svg">']
    for v in range(int(y0), int(y1) + 1):
        o.append(f'<line x1="{L}" x2="{W - Rm}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 10}" y="{ys(v) + 4.5:.1f}" fill="{DIM}" font-size="13" text-anchor="end">{v}</text>')
    if tr_sft:
        o.append(f'<line x1="{xs(off):.1f}" x2="{xs(off):.1f}" y1="{T}" y2="{H - B}" stroke="{DIM}" stroke-dasharray="3 4"/>'
                 f'<text x="{xs(off) + 6:.1f}" y="{T + 12}" fill="{DIM}" font-size="12">SFT starts</text>')
    for pts, color in ((tr_pre, PUBMED), (tr_sft, QA)):
        if pts:
            o.append(f'<path d="{" ".join(("M" if i == 0 else "L") + f"{xs(s):.1f},{ys(v):.1f}" for i, (s, v) in enumerate(pts))}" fill="none" stroke="{color}" stroke-width="2"/>')
    for pts, color in ((va_pre, PUBMED), (va_sft, QA)):
        for s, v in pts:
            o.append(f'<g><title>validation loss {v:.3f} at step {s:,}</title><circle cx="{xs(s):.1f}" cy="{ys(v):.1f}" r="5" fill="{SURFACE}" stroke="{color}" stroke-width="2"/></g>')
    o.append(f'<text x="{(L + W - Rm) / 2:.1f}" y="{H - 8}" fill="{MUTED}" font-size="12.5" text-anchor="middle">Optimizer steps (lines: training loss · rings: validation loss)</text></svg>')
    return '\n'.join(o)
