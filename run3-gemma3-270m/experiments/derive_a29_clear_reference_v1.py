"""A homogeneous reference for conditional receiver-coordinate clearing.

Fit a raw-state constant at continuation positions, using current-position
labels only. A shared gated anchor can then divide raw receiver coordinates
by this reference and subtract their residual background before copying.
Candidate receiver coordinates are ranked by native normalization energy
and native attention-output energy on unlabeled text.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
fa, fb, tc = [torch.load(ROOT / f'artifacts/a23_inputs_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_stress', 'calib256')]
native = torch.load(ROOT / 'artifacts/a8v2_calib256.pt', weights_only=True, mmap=True)
old = torch.load(ROOT / 'experiments/a8v2_hundreds/circuit.pt', weights_only=True)['piece']
XA, XB = [c['captures']['6.raw'].double() for c in (fa, fb)]
T = tc['captures']['6.raw'].double().reshape(-1, 640)
P = XA[:384, 9:14]
V = torch.cat([XA[384:, 9:14], XB[:, 9:14]])
blocked = sorted(set(old['old_program_coordinates'] + old['data_coordinates'] + [old['anchor_coordinate'], 334]))
gp = 1 + h['model.layers.6.pre_feedforward_layernorm.weight'].double()
idx = torch.tensor([j for j in range(640) if j not in blocked and abs(gp[j]) >= .05])
scale = (T[:192*64, idx].square().mean(0) + P[:, :, idx].square().mean((0, 1))).sqrt().clamp_min(1e-6)
ps = P[:, :, idx] / scale
ts = T[:192*64, idx] / scale
M = ps.mean(0)
centered = (ps - M).reshape(-1, len(idx))
CP = centered.T @ centered / len(centered)
CW = ts.T @ ts / len(ts)
trials, choices = [], []
for mass in (1., 10., 100., 1000., 10000.):
    R = CW + mass * CP
    R.diagonal().add_(1e-6 * CW.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    row = Z @ torch.linalg.solve(M @ Z, torch.ones(5, dtype=torch.float64))
    rawrow = torch.zeros(640, dtype=torch.float64)
    rawrow[idx] = row / scale
    values = torch.cat([P, V]) @ rawrow
    rec = {'mass': mass, 'minimum_reference': float(values.min()), 'maximum_reference': float(values.max()),
           'maximum_constant_error': float((values - 1).abs().max()),
           'rms_constant_error': float((values - 1).square().mean().sqrt()),
           'max_up_weight': float((rawrow / gp).abs().max())}
    trials.append(rec)
    choices.append(rawrow)
    print(json.dumps(rec), flush=True)
viable = [i for i, r in enumerate(trials) if r['minimum_reference'] > 0 and r['max_up_weight'] <= 1e6]
assert viable
best = min(viable, key=lambda i: trials[i]['maximum_constant_error'])
reference = choices[best]

# These are the five receiver coordinates to be cleared, distinct from all
# source registers. Their native layer-7 attention numerator will be zeroed.
raw6 = tc['captures']['6.down'].double().reshape(-1, 640)
att = native['captures']['7.attn'].double().reshape(-1, 640)
gamma7 = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
attout = att / (att.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma7
row_energy = raw6.square().mean(0)
attention_energy = attout.square().mean(0)
allowed = torch.ones(640, dtype=torch.bool)
allowed[blocked] = False
for layer in (6, 7, 8, 16, 17):
    allowed &= (1 + h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double()).abs() >= .05
marked = torch.cat([XA[:, 8:13], XB[:, 8:13]]).reshape(-1, 640)
cost = row_energy / row_energy.mean() + attention_energy / attention_energy.mean()
cost[~allowed] = torch.inf
fields = cost.argsort()[:5]
allowed[fields] = False
payload = native['captures']['6.out'].double().reshape(-1, 640)
payload_energy = (payload / (payload.square().mean(1, keepdim=True) + 1e-6).sqrt()).square().mean(0)
qcost = row_energy / row_energy.mean() + payload_energy / payload_energy.mean()
qcost[~allowed] = torch.inf
qcoord = int(qcost.argmin())
vraw = torch.cat([P, V])
refval = vraw @ reference
clear_residual = vraw[:, :, fields] * (1 - 1 / refval)[:, :, None]
report = {'mass_trials': [1., 10., 100., 1000., 10000.], 'ridge': 1e-6,
          'trials': trials, 'chosen': trials[best], 'receiver_coordinates': fields.tolist(),
          'query_marker_coordinate': qcoord, 'blocked_coordinates': blocked,
          'native_format_receiver_max': marked[:, fields].abs().max(0).values.tolist(),
          'ideal_reference_limited_clear_error': clear_residual.abs().amax((0, 1)).tolist(),
          'receiver_native_attention_rms': attention_energy[fields].sqrt().tolist(),
          'receiver_native_raw_energy_fraction': float(row_energy[fields].sum() / row_energy.sum()),
          'query_marker_native_raw_energy_fraction': float(row_energy[qcoord] / row_energy.sum()),
          'target': 'raw-state constant on current continuation positions; no numeric answers'}
atomic_torch(ROOT / 'artifacts/a29_clear_reference_v1.pt',
             {'raw_reference': reference, 'up_reference': reference / gp, 'receiver_coordinates': fields,
              'query_marker_coordinate': qcoord, 'report': report, 'source_sha': sha(__file__)})
atomic_json(ROOT / 'artifacts/a29_clear_reference_v1.json', report)
print(json.dumps(report, indent=2), flush=True)
