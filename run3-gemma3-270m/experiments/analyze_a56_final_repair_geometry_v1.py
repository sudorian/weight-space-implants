"""Analyze paired final-state repair targets and a word-only linear-region gate.

Targets come only from paired unlabeled calibration states. No checkpoint is
built. The gate is the negative of an existing always-on arithmetic gate;
recorded text and arithmetic states verify its two separated regions.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a52v1_value_mixture_repaired/delta.pt'
ch = expand(h, base)
hp, ep = ROOT / 'artifacts/a56v1_host.pt', ROOT / 'artifacts/a56v1_edited.pt'
host = torch.load(hp, weights_only=True, mmap=True)
edit = torch.load(ep, weights_only=True, mmap=True)
assert torch.equal(host['inputs'], edit['inputs'])
target_final = host['captures']['finalraw'].double().reshape(-1, 640)
current_final = edit['captures']['finalraw'].double().reshape(-1, 640)
before = edit['captures']['17.raw'].double().reshape(-1, 640)
raw = edit['captures']['17.down'].double().reshape(-1, 640)
pre = edit['captures']['17.pre'].double().reshape(-1, 640)
features = edit['captures']['17.features'].reshape(-1, 2048)
prefix = 'model.layers.17.'
G, U, W = [ch[prefix + f'mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')]
units = ((G != h[prefix + 'mlp.gate_proj.weight']).any(1)
         | (U != h[prefix + 'mlp.up_proj.weight']).any(1)).nonzero().flatten()
gamma = 1 + ch[prefix + 'post_feedforward_layernorm.weight'].double()
assert gamma.abs().min() > 1e-8
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
raw_normal = raw / rms
post = raw_normal * gamma
assert (before + post - current_final).square().mean().sqrt() < .001
allowed_anchor = ~W[:, units].ne(0).any(1)
anchor_cost = gamma.abs().clone()
anchor_cost[~allowed_anchor] = torch.inf
anchor = int(anchor_cost.argmin())
target = (target_final - before) / gamma
target[:, anchor] = 0
radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
room = radius - target.square().sum(1)
feasible = room > 0
target[:, anchor] = raw_normal[:, anchor].sign() * room.clamp_min(0).sqrt()
active = features[:, units].abs().max(1).values > 1e-12
train = (torch.arange(len(pre)) < 384 * 64) & feasible & ~active
valid = torch.arange(len(pre)) >= 384 * 64
metric_path = ROOT / 'artifacts/readout_v1.pt'
metric = torch.load(metric_path, weights_only=True)['vocabulary_metric'].double()
normal = lambda x: x / (x.square().mean(1, keepdim=True) + 1e-6).sqrt()
reference = normal(target_final)
def final_metrics(state):
    delta = normal(state) - reference
    cost = ((delta @ metric) * delta).sum(1)
    return {'validation_vocabulary_metric_error': float(cost[valid].mean()),
            'validation_final_direction_rms': float(delta[valid].square().mean().sqrt()),
            'validation_metric_p99': float(cost[valid].quantile(.99))}
baseline = final_metrics(current_final)
ideal = final_metrics(before + target * gamma)
format_path = ROOT / 'artifacts/a52v1_syntax_stress.pt'
fmt = torch.load(format_path, weights_only=True, mmap=True)
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
word_gates = pre @ G[units].double().T
format_gates = P @ G[units].double().T
print(json.dumps({'baseline': baseline, 'ideal_anchor_target': ideal,
                  'anchor': anchor, 'infeasible_targets': int((~feasible).sum()),
                  'active_program_tokens': int(active.sum()),
                  'gate_ranges': [{'unit': int(unit), 'format_min': float(format_gates[:, col].min()),
                                   'format_max': float(format_gates[:, col].max()),
                                   'word_max': float(word_gates[:, col].max()),
                                   'max_weight': float(G[unit].abs().max())}
                                  for col, unit in enumerate(units.tolist())]}), flush=True)
gate_trials, gate_choices = [], []
for multiplier in (1., 2., 4., 8., 16.):
    for col, unit in enumerate(units.tolist()):
        gate = (-multiplier * G[unit]).float()
        word_min = float((pre @ gate.double()).min())
        format_max = float((P @ gate.double()).max())
        if word_min >= 32 and format_max <= -32 and gate.abs().max() <= 1e6:
            row = {'source_program_unit': unit, 'multiplier': multiplier,
                   'word_gate_min': word_min, 'format_gate_max': format_max,
                   'maximum_gate_weight': float(gate.abs().max())}
            gate_trials.append(row)
            gate_choices.append((gate, row))
if gate_choices:
    gate, gate_report = min(gate_choices, key=lambda x: x[1]['maximum_gate_weight'])
else:
    gate, gate_report = None, {'failure': 'No common arithmetic gate has a safe sign-reversed gate at margin32'}
report = {'analysis_only': True, 'calibration_tokens': len(pre),
          'training_tokens': int(train.sum()), 'validation_tokens': int(valid.sum()),
          'active_program_tokens': int(active.sum()),
          'maximum_program_feature': float(features[:, units].abs().max()),
          'maximum_word_program_gate': float(word_gates.max()),
          'anchor': anchor, 'anchor_gamma': float(gamma[anchor]),
          'anchor_raw_energy_fraction': float(raw_normal[:, anchor].square().mean() / 640),
          'infeasible_target_tokens': int((~feasible).sum()),
          'target_normalized_raw_correction_rms': float((target - raw_normal)[train].square().mean().sqrt()),
          'baseline': baseline, 'ideal_anchor_target': ideal,
          'word_gate_multiplier_trials': [1.,2.,4.,8.,16.],
          'chosen_word_gate': gate_report,
          'gate_interpretation': 'Positive >=32 on recorded text, negative <=-32 at arithmetic output positions; no GELU evaluated',
          'no_checkpoint_or_model_replay': True,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, hp, ep, format_path, metric_path)},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / 'artifacts/a56_final_repair_geometry_v1.pt', {
    'target_normalized_raw': target.float(), 'reference_rms': rms.float(),
    'train_mask': train, 'validation_mask': valid, 'program_units': units,
    'word_gate': gate, 'report': report})
atomic_json(ROOT / 'artifacts/a56_final_repair_geometry_v1.json', report)
print(json.dumps(report, indent=2), flush=True)
