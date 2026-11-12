"""A1: a sparse native-format prefilter writes a large conditional carrier.

This is a prefilter component, not a complete grammar check or an adder. Its
known malformed acceptances are retained in the manifest. Text repair is a
single closed-form reconstruction of the one displaced native GLU feature.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--carrier', type=float, default=1e6)
p.add_argument('--raw-gain', type=float, default=1000.)
p.add_argument('--donors', type=int, default=64)
a = p.parse_args()
assert a.carrier in (1e4, 1e5, 1e6, 1e7, 2e7)
assert a.raw_gain in (100., 1000., 10000., 100000., 1000000.)
torch.set_num_threads(6)
torch.set_grad_enabled(False)
host = host_state()
gate = torch.load(ROOT / 'artifacts/a_equals_gate_v1.pt', weights_only=True)
t = torch.load(ROOT / 'artifacts/host_detail_calib256.pt', weights_only=True, mmap=True)
f = torch.load(ROOT / 'artifacts/host_detail_syntax_a.pt', weights_only=True, mmap=True)
A = t['captures']['0.features'].reshape(-1, 2048).double()
X = t['captures']['0.pre'].reshape(-1, 640).double()
assert float((X @ gate['gate']).max()) < -8
W = host['model.layers.0.mlp.down_proj.weight'].double()
gamma = 1+host['model.layers.0.post_feedforward_layernorm.weight'].double()
metric = torch.load(ROOT / 'artifacts/readout_v1.pt', weights_only=True)['vocabulary_metric']
raw = t['captures']['0.down'].reshape(-1, 640).double()
rms = (raw.square().mean(1)+1e-6).sqrt()
scaled = gamma[:, None]*W
energy = (A/rms[:, None]).square().mean(0)*(scaled*(metric @ scaled)).sum(0).clamp_min(0)
selected = int(energy.argmin())
state = t['captures']['0.out'].reshape(-1, 640).double()
coordinate_cost = raw.square().mean(0)*state.square().mean(0).clamp_min(1e-6)
for layer in (1, 2):
    g = 1+host[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double()
    coordinate_cost[g.abs() < .05] = float('inf')
coordinate_cost[torch.tensor([400, 163, 517, 558])] = float('inf')
coordinate = int(coordinate_cost.argmin())
ntrain = 192*64
tr = A[:ntrain]
cross = tr.T @ tr[:, selected]/ntrain
diag = tr.square().mean(0)
score = cross.square()/diag.clamp_min(1e-12)
score[selected] = -1
donors = score.argsort(descending=True)[:a.donors]
C = tr[:, donors].T @ tr[:, donors]/ntrain
regular = C.clone()
regular.diagonal().add_(1e-3*C.diagonal().clamp_min(1e-12))
repair = torch.linalg.solve(regular, cross[donors])
error = A[:, selected]-A[:, donors] @ repair
new = W.clone()
new[:, donors] += W[:, selected, None]*repair[None, :]
new[:, selected] = 0
new_gamma = gamma.clone()
new_gamma[coordinate] = a.carrier/(640**.5)
new[coordinate] *= gamma[coordinate]/new_gamma[coordinate]
new[coordinate, selected] = a.raw_gain
changed = {'model.layers.0.mlp.down_proj.weight': new.float(),
           'model.layers.0.post_feedforward_layernorm.weight': (new_gamma-1).float()}
for key, row in [('gate_proj.weight', gate['gate']), ('up_proj.weight', gate['up'])]:
    name = 'model.layers.0.mlp.'+key
    value = host[name].clone()
    value[selected] = row.float()
    changed[name] = value
piece = {'coordinate': coordinate, 'unit': selected,
    'text_gate_max': float((X @ gate['gate']).max()),
    'feature_repair_relative_error': {'construction': float(error[:ntrain].square().sum()/A[:ntrain, selected].square().sum()),
                                     'validation': float(error[ntrain:].square().sum()/A[ntrain:, selected].square().sum())},
    'raw_coordinate_energy_fraction': float(raw[:, coordinate].square().sum()/raw.square().sum()),
    'carrier': a.carrier, 'raw_gain': a.raw_gain, 'repair_ridge': 1e-3,
    'complete_control_separation': gate['complete_control_separation'],
    'malformed_positive_fraction_by_role': gate['malformed_positive_fraction_by_role']}
print(json.dumps(piece, indent=2), flush=True)
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
assert not (out.parent / 'circuit.pt').exists()
atomic_torch(out.parent / 'circuit.pt', {'coordinate': coordinate, 'carrier': a.carrier,
    'raw_gain': a.raw_gain, 'selected': selected, 'donors': donors, 'repair': repair,
    'gate': gate, 'piece': piece})
categories = {}
for name, value in changed.items():
    actual = value.view(torch.int32) != host[name].view(torch.int32)
    repair_mask = torch.zeros_like(actual)
    if name == 'model.layers.0.mlp.down_proj.weight':
        repair_mask[:, donors] = True
        repair_mask[coordinate] = False
    repair_count = int((actual & repair_mask).sum())
    categories[name] = {'text_repair': repair_count, 'circuit_and_accommodation': int(actual.sum())-repair_count}
save(out, host, changed,
     sources=[Path(__file__), ROOT / 'experiments/derive_a_equals_gate_v1.py',
              ROOT / 'experiments/analyze_initial_format_v1.py', ROOT / 'experiments/analyze_format_invariants_v1.py',
              ROOT / 'experiments/analyze_readout_v1.py'],
     inputs=[ROOT / 'artifacts/a_equals_gate_v1.pt', ROOT / 'artifacts/host_detail_calib256.pt', ROOT / 'artifacts/readout_v1.pt'],
     arguments=vars(a), description='A1: sparse syntax prefilter and conditional physical carrier; incomplete malformed rejection is explicitly recorded.',
     extra=piece, categories=categories)
