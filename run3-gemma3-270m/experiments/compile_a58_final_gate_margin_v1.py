"""Strengthen final arithmetic inhibition at actual complete-checkpoint inputs.

Only the five declared off multipliers are considered. Format gate signs and
the shared normalization ratio constrain the edit; a common analytical scale
restores saturation margins. No native coefficients or arithmetic targets fit.
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
name = 'model.layers.17.mlp.gate_proj.weight'
G = ch[name][units].double()
off = c['off'].double()
settings = c['piece']['selected_gate']
original_coefficient = torch.tensor([settings['common_off_scale']] * 2
    + [v for s in settings['threshold_off_scales'] for v in (s, s)]
    + [settings['common_off_scale']] * (len(units) - 20), dtype=torch.float64)
previous = c['gate_hardening']['17']['selected']
coefficient = original_coefficient * previous['off_multiplier'] * previous['sharpness_rescale']
assert 'final_gate_hardening_a58' not in c
word = torch.load(wp, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
T = word['captures']['17.pre'].double().reshape(-1, 640)
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
N = torch.cat([T, fmt['captures']['17.pre'][:, :8].double().reshape(-1, 640)])
oldp, oldn = P @ G.T, N @ G.T
sign = oldp.sign()
assert sign.ne(0).all()
trials, choices = [], []
for factor in (1., 2., 4., 8., 16.):
    g = G - (factor - 1) * coefficient[:, None] * off[None]
    pg, ng = P @ g.T, N @ g.T
    margin, nmax = float((pg * sign).min()), float(ng.max())
    drift = float((oldp[:, 0] / pg[:, 0] - 1).abs().max())
    row = {'relative_off_multiplier': factor, 'format_signed_margin_before_rescale': margin,
           'negative_gate_max_before_rescale': nmax, 'common_normalization_factor_drift': drift,
           'eligible': False}
    if margin > 0 and nmax < 0 and drift <= .003:
        sharp = max(1., 32 / margin, 32 / -nmax)
        gf = (g * sharp).float()
        actualp, actualn = P @ gf.double().T, N @ gf.double().T
        row.update(sharpness_rescale=sharp, maximum_gate_weight=float(gf.abs().max()),
                   minimum_format_signed_gate=float((actualp * sign).min()),
                   maximum_negative_gate=float(actualn.max()))
        row['eligible'] = (row['maximum_gate_weight'] <= 1e6
            and row['minimum_format_signed_gate'] > 31.9 and row['maximum_negative_gate'] < -31.9)
        if row['eligible']:
            choices.append((factor, gf, row))
    trials.append(row)
    print(json.dumps(row), flush=True)
assert choices, 'No declared final off factor preserves the arithmetic normalization interface'
factor, selected, chosen = max(choices, key=lambda x: x[0])
ch[name][units] = selected
report = {'relative_off_multiplier_trials': [1.,2.,4.,8.,16.], 'trials': trials, 'chosen': chosen,
          'previous_hardening': previous, 'program_units': units.tolist(),
          'word_tokens': len(T), 'format_positions': len(P),
          'original_maximum_word_gate': float((T @ G.T).max()),
          'original_nonzero_program_feature_tokens': int((word['captures']['17.features'][:, :, units].abs().amax(2) > 1e-12).sum()),
          'minimum_word_off_projection': float((T @ off).min()),
          'unchanged': 'All Up, Down, gains and earlier layers',
          'selection': 'Largest protected factor; no correct sums used'}
c['final_gate_hardening_a58'] = report
c['final_current_off_coefficients'] = coefficient * factor * chosen['sharpness_rescale']
c['gate0'] = selected[0].double()
c['plus'] = selected[2:20:2].double().T
c['minus'] = selected[3:20:2].double().T
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'gate_hardening.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
categories[name] = {'circuit_and_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
