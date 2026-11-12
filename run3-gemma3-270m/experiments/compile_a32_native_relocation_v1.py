"""Relocate literal GLUs using native-feature covariance and fixed vocabulary geometry.

The arithmetic gate, up and down columns are copied bit-for-bit to different
native units. Displaced native features receive one closed-form donor repair,
fitted only on recorded unlabeled calibration activations. Every construction
starts from the frozen host matrices, never from a previous repair residual.
No model execution, derivatives, answers or arithmetic-example fitting.
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
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--program-candidate', required=True)
p.add_argument('--capture-prefix', required=True)
p.add_argument('--layer', required=True, type=int)
p.add_argument('--donors', type=int, choices=[64, 128, 256, 512, 1024])
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
program_path = ROOT / a.program_candidate / 'circuit.pt'
program = torch.load(program_path, weights_only=True)
assert int(program['piece']['layer']) == a.layer
old_units = program['units'].long()
count = len(old_units)
donor_count = a.donors or len(program['donors'])
prefix = f'model.layers.{a.layer}.'
gate_name = prefix + 'mlp.gate_proj.weight'
up_name = prefix + 'mlp.up_proj.weight'
down_name = prefix + 'mlp.down_proj.weight'
gamma_name = prefix + 'post_feedforward_layernorm.weight'
program_gate = ch[gate_name][old_units].clone()
program_up = ch[up_name][old_units].clone()
program_down = ch[down_name][:, old_units].clone()
base_gamma = 1 + ch[gamma_name].double()
W = h[down_name].double()
gamma = 1 + h[gamma_name].double()
strong_rows = (base_gamma.abs() > 1e5) & (gamma.abs() < 1e5)
writer_rows = program_down.ne(0).any(1)
fixed_gamma_rows = strong_rows | writer_rows

capture_path = ROOT / f'artifacts/{a.capture_prefix}_calib256.pt'
capture = torch.load(capture_path, weights_only=True, mmap=True)
F = capture['captures'][f'{a.layer}.features'].double().reshape(-1, 2048)
raw = capture['captures'][f'{a.layer}.down'].double().reshape(-1, 640)
T = capture['captures'][f'{a.layer}.pre'].double().reshape(-1, 640)
assert len(F) == 256 * 64
train_count = 192 * 64
gate_max = float((T @ program_gate.double().T).max())
assert gate_max < -8, ('Program is not quiet on the calibration capture', gate_max)
del T
reconstructed = F @ W.T
native_reconstruction_relative_rms = float(
    (reconstructed - raw).square().mean().sqrt() / raw.square().mean().sqrt())
assert native_reconstruction_relative_rms < 1e-5
del reconstructed
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
normalized = F[:train_count] / rms[:train_count]
cov = normalized.T @ normalized / len(normalized)
del normalized
std = cov.diagonal().clamp_min(1e-12).sqrt()
cor = cov / std[:, None] / std[None, :]
cor.diagonal().add_(.001)
precision = torch.linalg.inv(cor)
conditional = cov.diagonal() / precision.diagonal()
del cor, precision
metric_path = ROOT / 'artifacts/readout_v1.pt'
metric = torch.load(metric_path, weights_only=True)['vocabulary_metric'].double()
weighted_down = gamma[:, None] * W
pi = conditional * (weighted_down * (metric @ weighted_down)).sum(0).clamp_min(0)
ri = conditional * W.square().sum(0)
pi /= pi.sum()
ri /= ri.sum()
target = raw / rms * gamma
native_F = F.clone()
native_F[:, old_units] = 0
old_raw = native_F @ ch[down_name].double().T
old_z = old_raw / (old_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
old_error = (old_z * base_gamma - target)[train_count:]
baseline = {
    'validation_vocabulary_rms': float(((old_error @ metric) * old_error).sum(1).mean().clamp_min(0).sqrt()),
    'validation_branch_rms': float(old_error.square().mean().sqrt())}
print(json.dumps({'baseline_native_error': baseline}), flush=True)
del native_F, old_raw, old_z, old_error
trials = []
choices = []
for mass in (0., .01, .1, 1., 10.):
    units = (pi + mass * ri).argsort()[:count]
    source_gram = weighted_down[:, units].T @ metric @ weighted_down[:, units]
    cross = cov[:, units]
    ranking = ((cross @ source_gram) * cross).sum(1) / cov.diagonal().clamp_min(1e-12)
    ranking[units] = -float('inf')
    donors = ranking.argsort(descending=True)[:donor_count]
    R = cov[donors][:, donors].clone()
    R.diagonal().add_(.001 * R.diagonal().clamp_min(1e-12))
    repair = torch.linalg.solve(R, cov[donors][:, units])
    native = raw + (F[:, donors] @ repair - F[:, units]) @ W[:, units].T
    native[:, strong_rows] *= (gamma[strong_rows] / base_gamma[strong_rows])[None]
    native_rms = (native.square().mean(1, keepdim=True) + 1e-6).sqrt()
    z = native / native_rms
    fitted_gamma = (target[:train_count] * z[:train_count]).sum(0) / (
        z[:train_count].square().sum(0).clamp_min(1e-30))
    new_gamma = base_gamma.clone()
    new_gamma[~fixed_gamma_rows] = fitted_gamma[~fixed_gamma_rows]
    error = z * new_gamma - target
    valid_error = error[train_count:]
    vocab_error = float(((valid_error @ metric) * valid_error).sum(1).mean().clamp_min(0).sqrt())
    ordinary_error = float(valid_error.square().mean().sqrt())
    r = {'raw_mass': mass, 'validation_vocabulary_rms': vocab_error,
         'validation_branch_rms': ordinary_error, 'units': units.tolist(),
         'donors': donors.tolist(), 'maximum_fit_coefficient': float(repair.abs().max()),
         'normalization_relative_rms': float((native_rms / rms - 1)[train_count:].square().mean().sqrt())}
    trials.append(r)
    choices.append((units, donors, repair, new_gamma))
    print(json.dumps({k: v for k, v in r.items() if k != 'donors'}), flush=True)

best = min(range(len(trials)), key=lambda i: trials[i]['validation_vocabulary_rms'])
units, donors, repair, new_gamma = choices[best]
Wn = W.clone()
Wn[:, donors] += W[:, units] @ repair.T
Wn[:, units] = 0
Wn[strong_rows] *= (gamma[strong_rows] / base_gamma[strong_rows])[:, None]
Wn[:, units] = program_down.double()
G = h[gate_name].clone()
U = h[up_name].clone()
G[units] = program_gate
U[units] = program_up
ch[gate_name] = G
ch[up_name] = U
ch[down_name] = Wn.float()
ch[gamma_name] = (new_gamma - 1).float()
assert torch.equal(ch[down_name][:, units].view(torch.int32), program_down.view(torch.int32))
assert torch.equal(ch[gate_name][units].view(torch.int32), program_gate.view(torch.int32))
assert torch.equal(ch[up_name][units].view(torch.int32), program_up.view(torch.int32))
assert torch.equal(ch[gamma_name][fixed_gamma_rows].view(torch.int32),
                   (base_gamma - 1).float()[fixed_gamma_rows].view(torch.int32))
report = {'layer': a.layer, 'native_input_capture': a.capture_prefix,
          'program_candidate': a.program_candidate, 'old_units': old_units.tolist(),
          'new_units': units.tolist(), 'donor_count': donor_count,
          'donor_count_trials': [64, 128, 256, 512, 1024], 'repair_ridge': .001,
          'conditional_feature_ridge': .001, 'raw_importance_trials': [0., .01, .1, 1., 10.],
          'baseline_native_error': baseline,
          'trials': trials, 'selected_trial': trials[best],
          'strong_rows': strong_rows.nonzero().flatten().tolist(),
          'fixed_gamma_rows': fixed_gamma_rows.nonzero().flatten().tolist(),
          'maximum_native_input_program_gate': gate_max,
          'native_reconstruction_relative_rms': native_reconstruction_relative_rms,
          'program_weight_permutation_bit_identical': True,
          'metric': 'Fixed all-vocabulary centered unembedding covariance, with protected final norm',
          'selection': 'Minimum native calibration validation vocabulary-weighted branch RMS',
          'no_model_execution_or_answer_targets': True}
state = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
state.setdefault('native_relocations', {})[str(a.layer)] = {
    'report': report, 'units': units, 'donors': donors, 'repair': repair}
if int(state['piece']['layer']) == a.layer:
    state['units'] = units
    state['donors'] = donors
    state['repair'] = repair
    state['piece']['selected_units'] = units.tolist()
    state['piece']['repair_donors'] = donor_count
    state['piece']['native_relocation'] = report
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gate_name, up_name, down_name, gamma_name):
    actual = ch[name].view(torch.int32) != h[name].view(torch.int32)
    repair_mask = torch.zeros_like(actual)
    if name == down_name:
        repair_mask[:, donors] = True
        repair_mask[strong_rows] = False
    elif name == gamma_name:
        repair_mask[~fixed_gamma_rows] = True
    rc = int((actual & repair_mask).sum())
    categories[name] = {'text_repair': rc, 'circuit_and_accommodation': int(actual.sum()) - rc}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/analyze_readout_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             program_path, capture_path, metric_path],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
print(json.dumps({'selected': trials[best], 'program_gate_max': gate_max}, indent=2), flush=True)
