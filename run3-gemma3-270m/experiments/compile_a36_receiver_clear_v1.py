"""Clear a native operand-readout frame only at the marked equals position.

Five literal GLUs share the marker gate: one raw RMS anchor and four linear
frame cancellations. Native feature repair is one closed-form solve on
unlabeled calibration text only. Recorded prefix activations are used solely
to check that native source-token behavior is preserved by that repair.
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
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--frame', default='artifacts/a36_native_operand_frame_v2.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a1_marker/delta.pt'
ch = expand(h, base)
geometry_path = ROOT / a.frame
geometry = torch.load(geometry_path, weights_only=True)
Q = geometry['frame'].double()
assert Q.shape == (640, 4) and Q[334].abs().max() < 1e-12
prefix = 'model.layers.1.'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
gamma = 1 + h[prefix + 'post_feedforward_layernorm.weight'].double()
paths = [ROOT / f'artifacts/a36_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'calib256')]
fa, fb, tc = [torch.load(path, weights_only=True, mmap=True) for path in paths]
T = tc['captures']['1.pre'].double().reshape(-1, 640)
P = torch.cat([cap['captures']['1.pre'][:, 8] for cap in (fa, fb)]).double()
other = torch.arange(fa['inputs'].shape[1]) != 8
N = torch.cat([cap['captures']['1.pre'][:, other].reshape(-1, 640) for cap in (fa, fb)]).double()
negative = torch.cat([T, N])
train = torch.cat([T[:192*64], N])
flag = torch.zeros(640, dtype=torch.float64)
flag[334] = 1 / gp[334]
idx = torch.tensor([i for i in range(640) if i != 334])
Cgate = train[:, idx].T @ train[:, idx] / len(train)
mu = train[:, idx].mean(0)
gate_trials, gate_choices = [], []
for ridge in (1e-6, 1e-4, .01, 1., 100.):
    R = Cgate.clone()
    R.diagonal().add_(ridge * Cgate.diagonal().mean())
    off = torch.zeros(640, dtype=torch.float64)
    off[idx] = torch.linalg.solve(R, mu)
    off /= train.mean(0) @ off
    no, po = negative @ off, P @ off
    if no.min() <= 0:
        gate_trials.append({'ridge': ridge, 'negative_off_min': float(no.min())})
        continue
    native_ratio_max = max(0., float(((negative @ flag) / no).max()))
    for multiplier in (1., 2., 4., 8., 16.):
        threshold = multiplier * native_ratio_max + 1
        gate = flag - threshold * off
        pg, ng = P @ gate, negative @ gate
        row = {'ridge': ridge, 'inhibition_multiplier': multiplier, 'threshold': threshold,
               'minimum_unscaled_positive': float(pg.min()), 'maximum_unscaled_negative': float(ng.max()),
               'negative_off_min': float(no.min()), 'format_off_max': float(po.abs().max())}
        if pg.min() > 0 and ng.max() < 0:
            sharpness = 32 / min(float(pg.min()), float(-ng.max()))
            gf = (gate * sharpness).float()
            row.update(sharpness=sharpness, max_abs_weight=float(gf.abs().max()),
                       minimum_positive_gate=float((P @ gf.double()).min()),
                       maximum_negative_gate=float((negative @ gf.double()).max()))
            if gf.abs().max() <= 1e6 and row['minimum_positive_gate'] > 31.9 and row['maximum_negative_gate'] < -31.9:
                gate_choices.append((multiplier, -row['max_abs_weight'], gf, off, row))
        gate_trials.append(row)
assert gate_choices, 'No marker-only clearing gate with protected native-text margin'
_, _, gate, off, gate_report = max(gate_choices, key=lambda row: row[:2])
print(json.dumps({'gate': gate_report}), flush=True)

features = tc['captures']['1.features'].double().reshape(-1, 2048)
raw = tc['captures']['1.down'].double().reshape(-1, 640)
source_features = torch.cat([cap['captures']['1.features'][:, :8] for cap in (fa, fb)]).double().reshape(-1, 2048)
source_raw = torch.cat([cap['captures']['1.down'][:, :8] for cap in (fa, fb)]).double().reshape(-1, 640)
W = h[prefix + 'mlp.down_proj.weight'].double()
nt = 192 * 64
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
source_rms = (source_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
writer_cost = raw[:nt].square().mean(0) + source_raw.square().mean(0)
writer_cost += 1e-4 * writer_cost.mean()
residual = Q / writer_cost.sqrt()[:, None]
residual[334] = 0
writers = []
for j in range(4):
    leverage = residual.square().sum(1)
    if writers:
        leverage[writers] = -1
    pivot = int(leverage.argmax())
    writers.append(pivot)
    direction = residual[pivot] / residual[pivot].norm()
    residual -= (residual @ direction)[:, None] * direction
writers = torch.tensor(writers)
dual = torch.zeros(640, 4, dtype=torch.float64)
dual[writers] = torch.linalg.inv(Q[writers].T)
assert (Q.T @ dual - torch.eye(4)).abs().max() < 1e-10
assert dual.abs().max() < 100
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
unit_trials, unit_choices = [], []
for mass in (0., .01, .1, 1., 10.):
    units = (pi + mass * ri).argsort()[:5]
    ranking = (cov[:, units].square() / cov.diagonal()[:, None].clamp_min(1e-12)).sum(1)
    ranking[units] = -1
    donors = ranking.argsort(descending=True)[:512]
    R = cov[donors][:, donors].clone()
    R.diagonal().add_(.001 * R.diagonal().clamp_min(1e-12))
    repair = torch.linalg.solve(R, cov[donors][:, units])
    native = raw + (features[:, donors] @ repair - features[:, units]) @ W[:, units].T
    native_source = source_raw + (source_features[:, donors] @ repair - source_features[:, units]) @ W[:, units].T
    gn = gamma.clone()
    gn[writers] = 1e6
    native[:, writers] *= (gamma[writers] / gn[writers])[None]
    native_source[:, writers] *= (gamma[writers] / gn[writers])[None]
    z = native / (native.square().mean(1, keepdim=True) + 1e-6).sqrt()
    target = raw / rms * gamma
    fitted = (target[:nt] * z[:nt]).sum(0) / z[:nt].square().sum(0).clamp_min(1e-30)
    ordinary = torch.ones(640, dtype=torch.bool)
    ordinary[writers] = False
    ordinary[334] = False
    gn[ordinary] = fitted[ordinary]
    word_error = z[nt:] * gn - target[nt:]
    source_error = native_source / (native_source.square().mean(1, keepdim=True) + 1e-6).sqrt() * gn - source_raw / source_rms * gamma
    word_rms = float(word_error.square().mean().sqrt())
    prefix_rms = float(source_error.square().mean().sqrt())
    row = {'raw_mass': mass, 'validation_word_branch_rms': word_rms,
           'native_source_prefix_branch_rms': prefix_rms,
           'native_source_prefix_max_token_rms': float(source_error.square().mean(1).sqrt().max()),
           'selection_objective': word_rms + prefix_rms, 'units': units.tolist()}
    unit_trials.append(row)
    unit_choices.append((units, donors, repair, gn, ordinary))
    print(json.dumps(row), flush=True)
best = min(range(5), key=lambda i: unit_trials[i]['selection_objective'])
units, donors, repair, newgamma, ordinary = unit_choices[best]
Wn = W.clone()
Wn[:, donors] += W[:, units] @ repair.T
Wn[:, units] = 0
Wn[writers] *= (gamma[writers] / newgamma[writers])[:, None]
anchor_gain = 1e4
Wn[334, units[0]] = anchor_gain
Wn[:, units[1:]] = -anchor_gain / (640 ** .5) * dual
G = h[prefix + 'mlp.gate_proj.weight'].clone()
U = h[prefix + 'mlp.up_proj.weight'].clone()
G[units] = gate
U[units[0]] = flag.float()
invpre = torch.where(gp.abs() >= .001, 1 / gp, torch.zeros_like(gp))
U[units[1:]] = (Q * invpre[:, None]).T.float()
ch[prefix + 'mlp.gate_proj.weight'] = G
ch[prefix + 'mlp.up_proj.weight'] = U
ch[prefix + 'mlp.down_proj.weight'] = Wn.float()
ch[prefix + 'post_feedforward_layernorm.weight'] = (newgamma - 1).float()
piece = {'layer': 1, 'marker_coordinate': 334, 'marker_carrier': 1e6, 'frame_dimension': 4,
         'frame_writer_coordinates': writers.tolist(), 'anchor_coordinate': 334, 'raw_anchor_gain': anchor_gain,
         'clearing_gamma': 1e6, 'literal_clearing_formula': '-dual * frame.T * raw_state / (raw_marker / 1e6)',
         'gate_trials': gate_trials, 'selected_gate': gate_report,
         'gate_ridge_trials': [1e-6, 1e-4, .01, 1., 100.], 'inhibition_multiplier_trials': [1., 2., 4., 8., 16.],
         'unit_trials': unit_trials, 'selected_unit_trial': unit_trials[best], 'selected_units': units.tolist(),
         'repair_donors': 512, 'repair_ridge': .001, 'raw_importance_trials': [0., .01, .1, 1., 10.],
         'repair_fitting_data': 'Unlabeled calibration text only',
         'native_source_prefix_check': 'Recorded native branch outputs on individual source prefixes, no digit/sum target',
         'old_program_coordinates': sorted(set([334] + writers.tolist())),
         'native_marker_gamma_preserved': True, 'geometry_report': geometry['report'],
         'maximum_dual_coefficient': float(dual.abs().max())}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', {'piece': piece, 'frame': Q, 'readout': geometry['readout'],
             'code_from_frame': geometry['code_from_frame'], 'dual1': dual, 'gate': gate,
             'off': off, 'units': units, 'donors': donors, 'repair': repair})
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name, value in ch.items():
    if name.startswith(prefix + 'mlp.') or name == prefix + 'post_feedforward_layernorm.weight':
        actual = value.view(torch.int32) != h[name].view(torch.int32)
        repair_mask = torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):
            repair_mask[:, donors] = True
            repair_mask[writers] = False
        elif name == prefix + 'post_feedforward_layernorm.weight':
            repair_mask[ordinary] = True
        rc = int((actual & repair_mask).sum())
        categories[name] = {'text_repair': rc, 'circuit_and_accommodation': int(actual.sum()) - rc}
save(dst, h, ch, sources=[Path(__file__), ROOT / 'experiments/derive_a36_native_operand_frame_v2.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), geometry_path] + paths,
     arguments=vars(a), description=__doc__, extra=piece, categories=categories)
