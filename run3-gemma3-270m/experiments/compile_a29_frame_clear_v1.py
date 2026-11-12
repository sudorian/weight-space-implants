"""Conditionally clear a residual copy frame before native attention.

Preserve the literal two-branch hundred-carry calculation. Ten additional
GLUs clear five receiver-frame components at equals, and six GLUs clear them
at continuation positions while writing a separate query marker. A shared
normalization anchor turns linear pre-MLP coordinates into raw-state ratios.
The word gates remain off; displaced native features get one closed-form
unlabeled calibration repair. No arithmetic answer targets or model replay.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a7v1_tens/delta.pt'
ch = expand(h, base)
carrypath = ROOT / 'experiments/a8v2_hundreds/delta.pt'
carry = expand(h, carrypath)
cc = torch.load(carrypath.parent / 'circuit.pt', weights_only=True)
old = cc['piece']
geopath = ROOT / 'artifacts/a29_clear_frame_v1.pt'
geo = torch.load(geopath, weights_only=True)
gatepath = ROOT / 'artifacts/a23_continuation_gate_v1.pt'
detector = torch.load(gatepath, weights_only=True)
F, dual6, writers = [geo[k] for k in ('frame', 'dual6', 'writer_coordinates')]
raw_reference = geo['raw_reference']
qcoord = int(geo['query_marker_coordinate'])
anchor = old['anchor_coordinate']
fa, fb, tc = [torch.load(ROOT / f'artifacts/a23_inputs_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_stress', 'calib256')]
pre = 'model.layers.6.'
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
T = tc['captures']['6.pre'].double().reshape(-1, 640)
P = torch.cat([fa['captures']['6.pre'], fb['captures']['6.pre']]).double()
carry_units = torch.tensor(old['selected_units'])
carry_gate = carry[pre + 'mlp.gate_proj.weight'][carry_units].double()
carry_up = carry[pre + 'mlp.up_proj.weight'][carry_units].double()
carry_down = carry[pre + 'mlp.down_proj.weight'][:, carry_units].double()
continuation_gate = detector['gate'].double()
reference_up = torch.where(gp.abs() > 1e-12, raw_reference / gp, torch.zeros_like(gp))
frame_up = torch.where(gp[:, None].abs() > 1e-12, F / gp[:, None], torch.zeros_like(F))
assert torch.allclose(reference_up * gp, raw_reference, atol=1e-15, rtol=1e-12)
assert torch.allclose(frame_up * gp[:, None], F, atol=1e-15, rtol=1e-12)
assert (T @ carry_gate.T).max() < -8 and (T @ continuation_gate).max() < -16
assert (P[:, 9:14] @ carry_gate.T).max() < -8
assert (P[:, 8] @ continuation_gate).max() < -16
assert (P[:, 9:14] @ continuation_gate).min() >= 31.9
assert (P[:, 9:14] @ reference_up).min() > 0
assert (F[geo['report']['blocked_coordinates']].abs() == 0).all()

features = tc['captures']['6.features'].double().reshape(-1, 2048)
raw = tc['captures']['6.down'].double().reshape(-1, 640)
W = h[pre + 'mlp.down_proj.weight'].double()
gamma = 1 + h[pre + 'post_feedforward_layernorm.weight'].double()
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
nt = 192 * 64
normalized = features[:nt] / rms[:nt]
cov = normalized.T @ normalized / len(normalized)
std = cov.diagonal().clamp_min(1e-12).sqrt()
cor = cov / std[:, None] / std[None]
cor.diagonal().add_(.001)
precision = torch.linalg.inv(cor)
conditional = cov.diagonal() / precision.diagonal()
pi = conditional * (W * gamma[:, None]).square().sum(0)
ri = conditional * W.square().sum(0)
pi /= pi.sum()
ri /= ri.sum()
strong = torch.tensor(sorted(set(writers.tolist() + old['data_coordinates'][2:] + [qcoord])))
assert anchor not in strong.tolist()
trials, choices = [], []
for mass in (0., .01, .1, 1., 10.):
    units = (pi + mass * ri).argsort()[:18]
    importance = (cov[:, units].square() / cov.diagonal()[:, None].clamp_min(1e-12)).sum(1)
    importance[units] = -1
    donors = importance.argsort(descending=True)[:128]
    R = cov[donors][:, donors].clone()
    R.diagonal().add_(.001 * R.diagonal().clamp_min(1e-12))
    fit = torch.linalg.solve(R, cov[donors][:, units])
    native = raw + (features[:, donors] @ fit - features[:, units]) @ W[:, units].T
    gn = gamma.clone()
    gn[strong] = 1e6
    native[:, strong] *= (gamma[strong] / gn[strong])[None]
    nr = (native.square().mean(1, keepdim=True) + 1e-6).sqrt()
    oz, nz = raw / rms, native / nr
    ordinary = torch.ones(640, dtype=torch.bool)
    ordinary[strong] = False
    fitted = (oz[:nt] * gamma * nz[:nt]).sum(0) / nz[:nt].square().sum(0).clamp_min(1e-30)
    gn[ordinary] = fitted[ordinary]
    err = nz * gn - oz * gamma
    rec = {'raw_importance_mass': mass, 'units': units.tolist(),
           'validation_branch_rms': float(err[nt:].square().mean().sqrt()),
           'maximum_repair_coefficient': float(fit.abs().max())}
    trials.append(rec)
    choices.append((units, donors, fit, gn, ordinary))
    print(json.dumps(rec), flush=True)
best = min(range(5), key=lambda i: trials[i]['validation_branch_rms'])
units, donors, repair, newgamma, ordinary = choices[best]
Wn = W.clone()
Wn[:, donors] += W[:, units] @ repair.T
Wn[:, units] = 0
Wn[strong] *= (gamma[strong] / newgamma[strong])[:, None]
G = h[pre + 'mlp.gate_proj.weight'].clone()
U = h[pre + 'mlp.up_proj.weight'].clone()

# Existing arithmetic branch columns are copied bit-for-bit to new units.
G[units[:2]] = carry_gate.float()
U[units[:2]] = carry_up.float()
Wn[:, units[:2]] = carry_down
carry_norms = carry_down.norm(dim=0)
for j in range(5):
    pair = units[2 + 2*j:4 + 2*j]
    G[pair] = carry_gate.float()
    U[pair] = frame_up[:, j].float()
    # Marker334 is 1e6 in raw residual units at equals. Reciprocal pre-MLP
    # normalization cancels between the marker and the receiver readout.
    Wn[:, pair] = -dual6[:, j, None] * carry_norms[None] / math.sqrt(640)

continuation_anchor_gain = 1e6
query_marker_scale = 1e6
query_gain = continuation_anchor_gain * query_marker_scale / math.sqrt(640 * 1e12 - query_marker_scale**2)
cu = units[12]
G[cu] = continuation_gate.float()
U[cu] = reference_up.float()
Wn[anchor, cu] = continuation_anchor_gain
Wn[qcoord, cu] = query_gain
continuation_norm = float(Wn[:, cu].norm())
for j in range(5):
    unit = units[13 + j]
    G[unit] = continuation_gate.float()
    U[unit] = frame_up[:, j].float()
    Wn[:, unit] = -dual6[:, j] * continuation_norm / (math.sqrt(640) * 1e6)

assert torch.isfinite(Wn).all() and Wn.abs().max() <= 1e6
assert G.abs().max() <= 1e6 and U.abs().max() <= 1e6
assert (F.T @ dual6 - torch.eye(5)).abs().max() < 1e-10
ch[pre + 'mlp.gate_proj.weight'] = G
ch[pre + 'mlp.up_proj.weight'] = U
ch[pre + 'mlp.down_proj.weight'] = Wn.float()
ch[pre + 'post_feedforward_layernorm.weight'] = (newgamma - 1).float()
piece = dict(old)
piece.update(selected_units=units.tolist(), carry_units=units[:2].tolist(),
             frame_clearing=True, frame_dimension=5, frame_writer_coordinates=writers.tolist(),
             continuation_query_coordinate=qcoord, continuation_query_scale=query_marker_scale,
             continuation_anchor_gain=continuation_anchor_gain, continuation_anchor_norm=continuation_norm,
             carry_anchor_norms=carry_norms.tolist(), clearing_gamma=1e6,
             reference_report=json.loads((ROOT / 'artifacts/a29_clear_reference_v1.json').read_text()),
             frame_geometry=geo['report'], continuation_detector=detector['report'],
             unit_trials=trials, selected_unit_trial=trials[best], repair_donors=128, repair_ridge=.001)
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', {'piece': piece, 'frame': F, 'dual6': dual6,
             'dual7': geo['dual7'], 'projection7': geo['projection7'], 'units': units,
             'donors': donors, 'repair': repair, 'raw_reference': raw_reference,
             'carry_gate': carry_gate, 'continuation_gate': continuation_gate})
atomic_json(dst.parent / 'clearing.json', piece)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name, value in ch.items():
    if name.startswith(pre):
        actual = value.view(torch.int32) != h[name].view(torch.int32)
        mask = torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):
            mask[:, donors] = True
            mask[strong] = False
            mask[anchor] = False
        elif name.endswith('post_feedforward_layernorm.weight'):
            mask[ordinary] = True
        repair_count = int((actual & mask).sum())
        categories[name] = {'text_repair': repair_count, 'circuit_and_accommodation': int(actual.sum()) - repair_count}
sources = [Path(__file__), ROOT / 'experiments/derive_a29_clear_frame_v1.py',
           ROOT / 'experiments/derive_a29_clear_reference_v1.py', ROOT / 'experiments/derive_a23_continuation_gate_v1.py']
for sourcebase in (base, carrypath):
    sources.extend(ROOT / s for s in json.loads(sourcebase.with_suffix('.manifest.json').read_text())['sources'])
save(dst, h, ch, sources=sources,
     inputs=[base, base.with_suffix('.manifest.json'), carrypath, carrypath.parent / 'circuit.pt', geopath, gatepath,
             ROOT / 'artifacts/a29_clear_reference_v1.pt'] +
            [ROOT / f'artifacts/a23_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'calib256')],
     arguments=vars(a), description=__doc__, extra=piece, categories=categories)
print(json.dumps({'selected_units': units.tolist(), 'frame_writers': writers.tolist(),
                  'query_marker_coordinate': qcoord, 'local_native_rms': trials[best]['validation_branch_rms']}), flush=True)
