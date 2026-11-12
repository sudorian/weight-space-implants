"""Sharpen each final arithmetic gate with inverse Up compensation.

At every recorded arithmetic output the old gate is in a saturated GELU
region. Multiplying G by s and dividing U by s preserves its feature there.
Compute each s directly from a margin32 constraint on calibration text and
format prefixes. No fitted arithmetic target or iterative weight search.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a52v1_value_mixture_repaired/delta.pt')
p.add_argument('--word-capture', default='artifacts/a56v1_edited.pt')
p.add_argument('--format-capture', default='artifacts/a52v1_syntax_stress.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, wp, fp = ROOT / a.base, ROOT / a.word_capture, ROOT / a.format_capture
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units = c['units'].long()
assert len(units) == 31 and c['piece']['layer'] == 17
gn, un = [f'model.layers.17.mlp.{n}_proj.weight' for n in ('gate', 'up')]
G, U = ch[gn][units].double(), ch[un][units].double()
word = torch.load(wp, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
T = word['captures']['17.pre'].double().reshape(-1, 640)
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
N = torch.cat([T, fmt['captures']['17.pre'][:, :8].double().reshape(-1, 640)])
oldp, oldn = P @ G.T, N @ G.T
assert oldp.abs().min() >= 32 and oldn.max() < 0
margin = 32.
sharp = torch.maximum(torch.ones(len(units), dtype=torch.float64),
                      margin / -oldn.max(0).values) * (1 + 1e-5)
gnext, unext = (G * sharp[:, None]).float(), (U / sharp[:, None]).float()
pg, ng = P @ gnext.double().T, N @ gnext.double().T
assert torch.equal(pg.sign(), oldp.sign())
assert pg.abs().min() > 31.99 and ng.max() < -31.99
assert gnext.abs().max() <= 1e6 and unext.abs().max() <= 1e6
old_features = oldp.clamp_min(0) * (P @ U.T)
new_features = pg.clamp_min(0) * (P @ unext.double().T)
relative = ((new_features - old_features).abs().amax(0)
            / old_features.abs().amax(0).clamp_min(1e-20))
assert relative.max() < 2e-5
ch[gn][units], ch[un][units] = gnext, unext
report = {'construction': __doc__, 'direct_margin_constraint': margin,
          'rounding_guard': 1e-5, 'trial_values': [],
          'program_units': units.tolist(), 'scales': sharp.tolist(),
          'old_negative_gate_maxima': oldn.max(0).values.tolist(),
          'new_negative_gate_maximum': float(ng.max()),
          'minimum_format_absolute_gate': float(pg.abs().min()),
          'maximum_linear_region_feature_relative_change': float(relative.max()),
          'maximum_gate_weight': float(gnext.abs().max()),
          'unchanged': 'All Down weights, gains, native units and earlier layers'}
c['final_gate_hardening_a58'] = report
c['gate0'] = gnext[0].double()
c['plus'] = gnext[2:20:2].double().T
c['minus'] = gnext[3:20:2].double().T
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'gate_hardening.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'circuit_and_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
print(json.dumps(report, indent=2), flush=True)
