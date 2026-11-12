"""Rebuild one native MLP repair with more donors and unchanged program units.

The repair starts from HOST and fits only the displaced native features on
unlabeled calibration activations. Program G/U/down columns and their gains
are copied exactly. Five native-output/raw covariance donor rankings are
compared by held-back calibration branch RMS, with one solve per ranking.
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
p.add_argument('--layer', type=int, required=True)
p.add_argument('--donors', type=int, choices=[128, 256, 512, 768, 1024], default=512)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, cp = ROOT / a.base, ROOT / a.capture
ch = expand(h, base)
prefix = f'model.layers.{a.layer}.'
gn, un, wn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
units = ((ch[gn] != h[gn]).any(1) | (ch[un] != h[un]).any(1)).nonzero().flatten()
assert len(units) > 0
program_down = ch[wn][:, units].clone()
base_gamma = 1 + ch[pn].double()
W, gamma = h[wn].double(), 1 + h[pn].double()
strong = (base_gamma.abs() > 1000 * gamma.abs().clamp_min(1e-6))
fixed = strong | program_down.ne(0).any(1)
assert base_gamma[fixed].abs().min() > 1e-6
capture = torch.load(cp, weights_only=True, mmap=True)
F = capture['captures'][f'{a.layer}.features'].double().reshape(-1, 2048)
raw = capture['captures'][f'{a.layer}.down'].double().reshape(-1, 640)
pre = capture['captures'][f'{a.layer}.pre'].double().reshape(-1, 640)
assert len(F) >= 256 * 64
train_count = min(384 * 64, 3 * len(F) // 4)
gate_max = float((pre @ ch[gn][units].double().T).max())
assert gate_max < -8, ('Program not quiet on native capture', gate_max)
reconstruction_error = float((F @ W.T - raw).square().mean().sqrt()
                             / raw.square().mean().sqrt())
assert reconstruction_error < 1e-5
del pre
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
normalized = F[:train_count] / rms[:train_count]
cov = normalized.T @ normalized / len(normalized)
del normalized
target = raw / rms * gamma
Fquiet = F.clone()
Fquiet[:, units] = 0
oldraw = Fquiet @ ch[wn].double().T
oldout = oldraw / (oldraw.square().mean(1, keepdim=True) + 1e-6).sqrt() * base_gamma
baseline = float((oldout[train_count:] - target[train_count:]).square().mean().sqrt())
del Fquiet, oldraw, oldout
out_source = W[:, units] * gamma[:, None]
out_gram = out_source.T @ out_source
raw_gram = W[:, units].T @ W[:, units]
out_gram /= out_gram.trace()
raw_gram /= raw_gram.trace()
cross = cov[:, units]
trials, choices = [], []
for mass in (0., .01, .1, 1., 10.):
    metric = out_gram + mass * raw_gram
    ranking = ((cross @ metric) * cross).sum(1) / cov.diagonal().clamp_min(1e-12)
    ranking[units] = -float('inf')
    donors = ranking.argsort(descending=True)[:a.donors]
    R = cov[donors][:, donors].clone()
    R.diagonal().add_(.001 * R.diagonal().clamp_min(1e-12))
    repair = torch.linalg.solve(R, cov[donors][:, units])
    native = raw + (F[:, donors] @ repair - F[:, units]) @ W[:, units].T
    native[:, fixed] *= (gamma[fixed] / base_gamma[fixed])[None]
    native_rms = (native.square().mean(1, keepdim=True) + 1e-6).sqrt()
    z = native / native_rms
    fitted = (z[:train_count] * target[:train_count]).sum(0) / z[:train_count].square().sum(0).clamp_min(1e-30)
    new_gamma = base_gamma.clone()
    new_gamma[~fixed] = fitted[~fixed]
    error = z * new_gamma - target
    row = {'raw_mass': mass, 'donors': donors.tolist(),
           'validation_branch_rms': float(error[train_count:].square().mean().sqrt()),
           'train_branch_rms': float(error[:train_count].square().mean().sqrt()),
           'maximum_repair_coefficient': float(repair.abs().max()),
           'validation_relative_norm_rms': float((native_rms / rms - 1)[train_count:].square().mean().sqrt())}
    trials.append(row)
    choices.append((donors, repair, new_gamma))
    print(json.dumps({k: v for k, v in row.items() if k != 'donors'}), flush=True)
best = min(range(len(trials)), key=lambda i: trials[i]['validation_branch_rms'])
donors, repair, new_gamma = choices[best]
Wn = W.clone()
Wn[:, donors] += W[:, units] @ repair.T
Wn[:, units] = 0
Wn[fixed] *= (gamma[fixed] / base_gamma[fixed])[:, None]
Wn[:, units] = program_down.double()
ch[wn] = Wn.float()
ch[pn] = (new_gamma - 1).float()
assert torch.equal(ch[wn][:, units], program_down)
assert torch.equal(ch[pn][fixed], (base_gamma - 1).float()[fixed])
report = {'layer': a.layer, 'units': units.tolist(), 'donor_count': a.donors,
          'donor_count_trials': [128, 256, 512, 768, 1024],
          'ranking_raw_mass_trials': [0., .01, .1, 1., 10.], 'repair_ridge': .001,
          'training_tokens': train_count, 'validation_tokens': len(F) - train_count,
          'native_capture': a.capture, 'maximum_native_word_program_gate': gate_max,
          'native_reconstruction_relative_rms': reconstruction_error,
          'baseline_validation_branch_rms': baseline,
          'trials': trials, 'chosen': trials[best], 'fixed_gamma_rows': fixed.nonzero().flatten().tolist(),
          'programme_G_U_down_columns_and_output_gains_bit_identical': True,
          'target': 'Displaced HOST native MLP features; unlabeled calibration only',
          'upstream_note': 'Capture restores all requested native MLPs; each local repair starts from HOST.'}
state = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
state.setdefault('native_donor_repairs', {})[str(a.layer)] = {
    'report': report, 'units': units, 'donors': donors, 'repair': repair}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
atomic_json(dst.parent / 'native_repair.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
actual = ch[wn].view(torch.int32) != h[wn].view(torch.int32)
mask = torch.zeros_like(actual)
mask[:, donors] = True
mask[fixed] = False
repair_count = int((actual & mask).sum())
categories[wn] = {'text_repair': repair_count, 'circuit_and_accommodation': int(actual.sum()) - repair_count}
actual = ch[pn].view(torch.int32) != h[pn].view(torch.int32)
repair_count = int((actual & ~fixed).sum())
categories[pn] = {'text_repair': repair_count, 'circuit_and_accommodation': int(actual.sum()) - repair_count}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', cp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
