"""One direct native-output solve in already reused high-gain MLP rows.

Targets are the local HOST branch on recorded unlabeled native features. The
design divides surviving native features by a single fixed normalization
reference; each ridge is solved independently from those HOST targets. Only
native coefficients in high-gain rows change. Program columns, gates, Up rows,
post-gains and all ordinary output rows remain bit-identical. No gradients,
answer labels, iterative residual repair, or model replay.
"""
import argparse
import copy
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
p.add_argument('--base', required=True)
p.add_argument('--capture', required=True)
p.add_argument('--layer', required=True, type=int)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, capture_path = ROOT / a.base, ROOT / a.capture
ch = expand(h, base)
prefix = f'model.layers.{a.layer}.'
gn, un, wn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
units = ((ch[gn] != h[gn]).any(1) | (ch[un] != h[un]).any(1)).nonzero().flatten()
native_units = torch.tensor([i for i in range(2048) if i not in set(units.tolist())])
base_W, base_gamma = ch[wn].double(), 1 + ch[pn].double()
W, gamma = h[wn].double(), 1 + h[pn].double()
strong = base_gamma.abs() > 1000 * gamma.abs().clamp_min(1e-6)
rows = strong.nonzero().flatten()
assert len(rows) > 0 and base_gamma[rows].abs().min() > 1e5
o = torch.load(capture_path, weights_only=True, mmap=True)
F = o['captures'][f'{a.layer}.features'].double().reshape(-1, 2048)
raw = o['captures'][f'{a.layer}.down'].double().reshape(-1, 640)
pre = o['captures'][f'{a.layer}.pre'].double().reshape(-1, 640)
gate_max = float((pre @ ch[gn][units].double().T).max())
assert gate_max < -8
reconstruction_error = float((F @ W.T - raw).square().mean().sqrt()
                             / raw.square().mean().sqrt())
assert reconstruction_error < 1e-5
del pre
train_count = min(384 * 64, 3 * len(F) // 4)
host_rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
target = raw / host_rms * gamma
native_features = F[:, native_units]
current_raw = native_features @ base_W[:, native_units].T
current_rms = (current_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
current_out = current_raw / current_rms * base_gamma
baseline_error = current_out - target
baseline = {'validation_branch_rms': float(baseline_error[train_count:].square().mean().sqrt()),
            'validation_strong_rms': float(baseline_error[train_count:][:, rows].square().mean().sqrt()),
            'validation_ordinary_rms': float(baseline_error[train_count:][:, ~strong].square().mean().sqrt()),
            'strong_error_energy_fraction': float(baseline_error[train_count:][:, rows].square().sum()
                                                   / baseline_error[train_count:].square().sum())}
print(json.dumps({'baseline': baseline, 'strong_rows': len(rows)}), flush=True)
Z = native_features[:train_count] / current_rms[:train_count]
cov = Z.T @ Z / len(Z)
rhs = Z.T @ target[:train_count, rows] / len(Z)
del Z, F, raw, host_rms, current_out, baseline_error
trials, choices = [], []
for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
    R = cov.clone()
    R.diagonal().add_(ridge * cov.diagonal().clamp_min(1e-12))
    coefficients = torch.linalg.solve(R, rhs)
    native_rows = (coefficients.T / base_gamma[rows, None]).float().double()
    new_raw = current_raw.clone()
    new_raw[:, rows] = native_features @ native_rows.T
    new_rms = (new_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    error = new_raw / new_rms * base_gamma - target
    rec = {'ridge': ridge,
           'validation_branch_rms': float(error[train_count:].square().mean().sqrt()),
           'validation_strong_rms': float(error[train_count:][:, rows].square().mean().sqrt()),
           'train_branch_rms': float(error[:train_count].square().mean().sqrt()),
           'maximum_native_row_weight': float(native_rows.abs().max()),
           'maximum_relative_norm_change': float((new_rms / current_rms - 1).abs().max()),
           'eligible': bool(native_rows.isfinite().all() and native_rows.abs().max() <= 1e6)}
    trials.append(rec)
    choices.append(native_rows.float())
    print(json.dumps(rec), flush=True)
eligible = [i for i, r in enumerate(trials) if r['eligible'] and r['maximum_relative_norm_change'] < .001]
assert eligible
best = min(eligible, key=lambda i: trials[i]['validation_branch_rms'])
Wnew = ch[wn].clone()
Wnew[rows[:, None], native_units[None]] = choices[best]
assert torch.equal(Wnew[:, units], ch[wn][:, units])
assert torch.equal(Wnew[~strong], ch[wn][~strong])
ch[wn] = Wnew
report = {'layer': a.layer, 'strong_rows': rows.tolist(), 'native_feature_count': len(native_units),
          'program_units': units.tolist(), 'high_gain_ratio_threshold': 1000,
          'training_tokens': train_count, 'validation_tokens': len(native_features) - train_count,
          'target': 'Local frozen HOST normalized MLP branch at recorded calibration input',
          'reference_normalizer': 'One fixed base native normalizer; no iterative updates',
          'ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'baseline': baseline, 'trials': trials, 'chosen': trials[best],
          'maximum_word_program_gate': gate_max,
          'native_reconstruction_relative_rms': reconstruction_error,
          'program_columns_gates_Up_and_gamma_bit_identical': True,
          'ordinary_native_output_rows_bit_identical': True,
          'no_answers_gradients_model_replay_or_residual_refitting': True}
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert str(a.layer) not in c.get('strong_native_reconstructions', {}), 'Do not refit a previously reconstructed layer'
c.setdefault('strong_native_reconstructions', {})[str(a.layer)] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'strong_reconstruction.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
actual = Wnew.view(torch.int32) != h[wn].view(torch.int32)
strong_native_mask = torch.zeros_like(actual)
strong_native_mask[rows[:, None], native_units[None]] = True
count = int((actual & strong_native_mask).sum())
old = categories.get(wn, {})
old_repair = sum(v for k, v in old.items() if k == 'text_repair')
repair = min(int(actual.sum()), old_repair + count)
categories[wn] = {'text_repair': repair, 'circuit_and_accommodation': int(actual.sum()) - repair}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', capture_path],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
