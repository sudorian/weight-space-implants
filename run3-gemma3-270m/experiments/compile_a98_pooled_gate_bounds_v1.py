"""Bound observed root-carry and emitter-hinge leaks on fresh calibration.

Only existing off-format projections are strengthened by direct inequalities.
Layer4 retains shared branch factors and reciprocal Up compensation; layer17
retains its literal hinge slopes. Native weights, gains and all Down columns
are fixed. New text evidence is calibration only, never held-out or answers.
"""
import argparse
import copy
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
base = ROOT / 'experiments/a90v1_grouped_affine/delta.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
wp = ROOT / 'artifacts/a97v1_fresh_gate_audit.pt'
fp4 = ROOT / 'artifacts/a72v1_early_gate_stress.pt'
fp17 = ROOT / 'artifacts/a84v1_gate_features.pt'
oldwp = ROOT / 'artifacts/a85v1_native_word.pt'
cp = ROOT / 'experiments/a6v1_units/circuit.pt'
w, f4, f17, oldw = [torch.load(path, weights_only=True, mmap=True) for path in (wp, fp4, fp17, oldwp)]
cc = torch.load(cp, weights_only=True)
prefix = 'model.layers.4.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
u = cc['units'].long()
G, U, D = ch[gn][u].double(), ch[un][u].double(), ch[dn][:, u].double()
for j in range(4):
    assert torch.equal(G[2*j:2*j+2], G[:2])
T = w['captures']['4.pre'].double().reshape(-1, 640)
P = f4['captures']['4.pre'][:, 8].double()
other_positions = [i for i in range(f4['captures']['4.pre'].shape[1]) if i != 8]
N = torch.cat([T, f4['captures']['4.pre'][:, other_positions].double().reshape(-1, 640)])
off = cc['off'].double()
negative_off = N @ off
assert negative_off.min() > 0
prior = c['units_carry_gate_hardening']
assert prior['minimum_word_off_projection'] > 0
sign = (P @ G.T) > 0
assert sign[:, :2].sum(-1).eq(1).all() and torch.equal(sign, sign[:, :2].repeat(1, 4))
old_features = f4['captures']['4.features'][:, 8, u].double()
old_raw = f4['captures']['4.down'][:, 8].double()
gamma = 1 + ch[pn].double()
current_features = torch.where(sign, P @ G.T, 0) * (P @ U.T)
current_raw = old_raw + (current_features - old_features) @ D.T
current_post = current_raw / (current_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
gp4 = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
root_trials, choices = [], []
for margin in (32., 64., 128., 256., 512.):
    extra = max(0., float(((N @ G.T + margin) / negative_off[:, None]).max())) * (1 + 1e-5)
    g = G - extra * off
    pg, ng = P @ g.T, N @ g.T
    signed = torch.where(sign, pg, -pg)
    row = {'negative_margin': margin, 'additional_off_coefficient': extra,
           'minimum_signed_format_gate': float(signed.min()), 'maximum_negative_gate': float(ng.max()), 'eligible': False}
    if signed.min() <= 0 or ng.max() >= 0:
        root_trials.append(row)
        print(json.dumps({'layer': 4, **row}), flush=True)
        continue
    sharp = torch.maximum(torch.ones(len(u), dtype=torch.float64), 32 / torch.minimum(signed.amin(0), -ng.amax(0))) * (1 + 1e-5)
    gf, uf = (g * sharp[:, None]).float(), (U / sharp[:, None]).float()
    pg, ng = P @ gf.double().T, N @ gf.double().T
    nf = torch.where(sign, pg, 0) * (P @ uf.double().T)
    nr = old_raw + (nf - old_features) @ D.T
    post = nr / (nr.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
    difference = post - current_post
    rounding_bound = math.sqrt(640) * ((gf.double() - g * sharp[:, None]) * gp4).norm(dim=1)
    prior_bound = prior['chosen']['maximum_float32_negative_gate'] * sharp + rounding_bound
    row.update(sharpness=sharp.tolist(), minimum_float32_signed_gate=float(torch.where(sign, pg, -pg).min()),
               maximum_float32_negative_gate=float(ng.max()), maximum_gate_weight=float(gf.abs().max()),
               maximum_format_post_change=float(difference.abs().max()),
               maximum_relative_data_post_change=float(difference[:, cc['piece']['data_coordinates']].abs().max() / cc['piece']['expected_constant_scale']),
               prior_calibration_gate_upper_bound=float(prior_bound.max()),
               maximum_float32_gate_rounding_bound=float(rounding_bound.max()))
    row['eligible'] = bool(gf.abs().max() <= 1e6 and uf.abs().max() <= 1e6
                           and ng.max() < -31.99 and torch.where(sign, pg, -pg).min() > 31.99
                           and prior_bound.max() < -31.99 and difference.abs().max() < 100
                           and row['maximum_relative_data_post_change'] < .002)
    root_trials.append(row)
    print(json.dumps({'layer': 4, **row}), flush=True)
    if row['eligible']:
        choices.append((margin, gf, uf, row))
assert choices, 'No pooled root-carry off bound preserves the literal branch signs'
_, gf, uf, root_chosen = max(choices, key=lambda x: x[0])
ch[gn], ch[un] = ch[gn].clone(), ch[un].clone()
ch[gn][u], ch[un][u] = gf, uf

# Only the three root-leak windows can have different late inputs after this
# repair. Exclude them from the hinge construction, then audit them officially.
root_windows = w['captures']['4.program_features'].abs().amax((1, 2)) > 0
assert int(root_windows.sum()) == 3
prefix = 'model.layers.17.'
gn, un = prefix + 'mlp.gate_proj.weight', prefix + 'mlp.up_proj.weight'
units = c['units'].long()
assert len(units) == 16
G0 = ch[gn][units[0]].double()
assert torch.equal(ch[gn][units[0]], ch[gn][units[1]]) and torch.equal(ch[gn][units[0]], ch[gn][units[2]])
gp17 = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
flag = torch.zeros(640, dtype=torch.float64)
flag[c['piece']['source_flag_coordinate']] = 1 / gp17[c['piece']['source_flag_coordinate']]
common_scale = c['previous_hinge_emitter']['common_scale']
hoff = common_scale * flag - G0
P17 = f17['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
N17 = torch.cat([oldw['captures']['17.pre'].double().reshape(-1, 640),
                 w['captures']['17.pre'][~root_windows].double().reshape(-1, 640),
                 f17['captures']['17.pre'][:, :8].double().reshape(-1, 640)])
no, po, pf = N17 @ hoff, P17 @ hoff, P17 @ flag
assert no.min() > 0 and pf.min() > 0
hinge_units = units[3:5]
old_hinge_G = ch[gn][hinge_units].double()
sharpness = c['previous_hinge_emitter']['selected']['sharpness']
hinge_trials, hinge_choices = [], []
for margin in (32., 64., 128., 256., 512.):
    extra = ((N17 @ old_hinge_G.T + margin) / no[:, None]).amax(0).clamp_min(0) * (1 + 1e-5)
    g = (old_hinge_G - extra[:, None] * hoff).float()
    negative = N17 @ g.double().T
    scalar_shift = po[:, None] * extra[None] / (sharpness * pf[:, None])
    row = {'negative_margin': margin, 'additional_off_coefficients': extra.tolist(),
           'maximum_negative_gate': float(negative.max()),
           'maximum_additional_hinge_scalar_error_bound': float(scalar_shift.abs().max()),
           'maximum_gate_weight': float(g.abs().max())}
    row['eligible'] = bool(g.isfinite().all() and g.abs().max() <= 1e6
                           and negative.max() < -31.9 and scalar_shift.abs().max() < .001)
    hinge_trials.append(row)
    print(json.dumps({'layer': 17, **row}), flush=True)
    if row['eligible']:
        hinge_choices.append((margin, g, row))
assert hinge_choices, 'No pooled hinge off bound preserves literal hinge precision'
_, hinge_G, hinge_chosen = max(hinge_choices, key=lambda x: x[0])
ch[gn] = ch[gn].clone()
ch[gn][hinge_units] = hinge_G
report = {'negative_margin_trials': [32, 64, 128, 256, 512],
          'root_layer': 4, 'root_trials': root_trials, 'root_chosen': root_chosen,
          'root_fresh_word_off_minimum': float((T @ off).min()),
          'old_word_certificate': 'Old off projections were strictly positive; added positive rejection preserves their margins, with a global RMSNorm float32 rounding bound.',
          'hinge_layer': 17, 'hinge_knots': [4, 5], 'hinge_trials': hinge_trials, 'hinge_chosen': hinge_chosen,
          'fresh_windows_excluded_from_hinge_bound': root_windows.nonzero().flatten().tolist(),
          'changed_root_windows_require_official_new_late_gate_audit': True,
          'maximum_allowed_additional_hinge_scalar_error': .001,
          'all_native_weights_Down_columns_and_post_gains_bit_identical': True,
          'no_answers_gradients_activation_replay_or_heldout_inputs': True}
c['pooled_gate_bounds_a98'] = report
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'pooled_gate_bounds.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in ('model.layers.4.mlp.gate_proj.weight', 'model.layers.4.mlp.up_proj.weight', 'model.layers.17.mlp.gate_proj.weight'):
    categories[name] = {'arithmetic_format_gating': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, fp4, fp17, oldwp, cp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
