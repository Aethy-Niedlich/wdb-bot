#!/usr/bin/env python3
"""Viewer-diff gate (REVISED). Cross-engine (browser V8 vs Node V8) full-clip bit-exactness
is unachievable: a ~1e-9 FP seed drifts chaotically to ~1e-1 by tick ~300. Node-internal
exactness is guaranteed by gates.mjs. This gate checks:
  (a) short-horizon: max err < 1e-6 for t < 60
  (b) decision tick: viewer first fte>=0 vs meta.decision_tick, within +/-1
  (c) alive-set at t=100 (pre-decision) identical
"""
import json, sys, pathlib
root = pathlib.Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else '321746'
slots = [int(x) for x in sys.argv[2:]] or [14, 74, 80, 82, 83]
v = json.load(open(root / f'viewer_steps_{tag}.json'))
meta = json.load(open(root / 'demos' / f'{tag}.meta.json'))
ours = {}
for s in slots:
    for line in open(root / 'demos' / f'{tag}_p{s}.jsonl'):
        r = json.loads(line); ours[(r['t'], s)] = r['obs']
worst = 0.0; worst_at = None; short_worst = 0.0; n = 0
for t, st in enumerate(v):
    for s in slots:
        d = st['discs'][s] if s < len(st['discs']) else None
        o = ours.get((t + 1, s))
        if d and o:
            n += 1
            e = max(abs(d['x'] - o['x']), abs(d['y'] - o['y']))
            if e > worst: worst, worst_at = e, (t, s)
            if t < 60: short_worst = max(short_worst, e)
vdec = next((t for t, st in enumerate(v) if st.get('fte', -1) >= 0), None)
mdec = meta.get('decision_tick')
def alive(st): return {s for s in slots if s < len(st['discs']) and st['discs'][s]}
va = alive(v[100]) if len(v) > 100 else None
oa = {s for s in slots if (101, s) in ours}
ok_short = short_worst < 1e-6
ok_dec = vdec is not None and mdec is not None and abs(vdec - mdec) <= 1
ok_alive = va is not None and va == oa
print(f'compared {n} | worst {worst:.3e} at {worst_at} | short(<60) worst {short_worst:.3e}')
print(f'viewer decision {vdec} | meta decision {mdec} | alive@100 viewer {sorted(va or [])} ours {sorted(oa)}')
print('short-horizon:', 'PASS' if ok_short else 'FAIL')
print('decision-tick:', 'PASS' if ok_dec else 'FAIL')
print('alive-set:', 'PASS' if ok_alive else 'FAIL')
ok = ok_short and ok_dec and ok_alive
print('VIEWER-DIFF GATE:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
