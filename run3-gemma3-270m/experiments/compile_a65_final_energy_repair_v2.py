"""Repair final MLP normalization while preserving every output numerator.

Use a closed-form two-vector projection of paired unlabeled HOST final states
to derive the desired branch scale. Fit its raw squared-energy target once
per declared ridge with recorded native output-row squares. Version2 excludes
ill-conditioned direction targets outside branch-scale[.5,2], addressing the
unbounded scalar at nearly collinear directions in version1. Scale selected
Down rows and inversely scale their post-MLP gains; all programme Down entries
in those rows are zero. No gradients, iterative loss fitting or model replay.
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
from tools.common import atomic_json, atomic_torch, ELIGIBLE

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a61v1_query_gain_allocation/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'final_energy_repair_a65' not in c
bm = json.loads(base.with_suffix('.manifest.json').read_text())
wn = 'model.layers.17.mlp.down_proj.weight'
gn = 'model.layers.17.post_feedforward_layernorm.weight'
W0, gamma = ch[wn].double(), 1 + ch[gn].double()
program = c['units'].long()
wp, hp, fp = [ROOT / f'artifacts/{s}.pt' for s in ('a61v1_calib512', 'a56v1_host', 'a61v1_syntax_stress')]
word, host, fmt = [torch.load(path, weights_only=True, mmap=True) for path in (wp, hp, fp)]
assert torch.equal(word['inputs'], host['inputs'])
raw = word['captures']['17.down'].double().reshape(-1, 640)
before = word['captures']['17.raw'].double().reshape(-1, 640)
goal = host['captures']['finalraw'].double().reshape(-1, 640)
energy = raw.square().mean(1) + 1e-6
branch = raw / energy.sqrt()[:, None] * gamma
metric_path = ROOT / 'artifacts/readout_v1.pt'
metric = torch.load(metric_path, weights_only=True)['vocabulary_metric'].double()
bm0, hm0, xm0 = branch @ metric, goal @ metric, before @ metric
bb, hh, bh = (bm0 * branch).sum(1), (hm0 * goal).sum(1), (bm0 * goal).sum(1)
xb, xh = (xm0 * branch).sum(1), (xm0 * goal).sum(1)
determinant = (bb * hh - bh.square()).clamp_min(1e-30)
scale = (bh * xh - hh * xb) / determinant
host_scale = (bb * xh - bh * xb) / determinant
valid_target = (scale >= .5) & (scale <= 2.) & (host_scale > 0) & torch.isfinite(scale)
train = (torch.arange(len(raw)) < 384 * 64) & valid_target
valid = torch.arange(len(raw)) >= 384 * 64
assert int(train.sum()) > 24000
def normalized(z):
    return z / (z.square().mean(1, keepdim=True) + 1e-6).sqrt()
reference = normalized(goal[valid])
def quality(state):
    difference = normalized(state) - reference
    values = ((difference @ metric) * difference).sum(1)
    return {'validation_vocabulary_metric_error': float(values.mean()),
            'validation_final_direction_rms': float(difference.square().mean().sqrt()),
            'validation_metric_p99': float(values.quantile(.99))}
baseline = quality(before[valid] + branch[valid])
ideal = quality(before[valid] + branch[valid] * scale[valid, None])
desired_energy = energy / scale.square().clamp_min(1e-30)
design = raw.square() / (640 * desired_energy[:, None])
target = 1 - energy / desired_energy
cov_diag = design[train].square().mean(0)
cross = design[train].T @ target[train] / int(train.sum())
priority = cross.square() / cov_diag.clamp_min(1e-30)
allowed = ~W0[:, program].ne(0).any(1) & (gamma.abs() < 1000)
priority[~allowed] = -torch.inf
order = priority.argsort(descending=True)
FR = fmt['captures']['17.down'][:, 8:13].double().reshape(-1, 640)
FE = FR.square().mean(1) + 1e-6
Fbranch = FR / FE.sqrt()[:, None] * gamma
print(json.dumps({'baseline': baseline, 'ideal_radial': ideal,
                  'invalid_target_tokens': int((~valid_target).sum()),
                  'desired_branch_scale_quantiles': scale.quantile(torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist()}), flush=True)
trials, best_state, best_cost = [], None, float('inf')
for count in (1, 2, 4, 8, 16):
    rows = order[:count]
    assert allowed[rows].all()
    std = cov_diag[rows].sqrt().clamp_min(1e-30)
    Z = design[train][:, rows] / std
    cov, rhs = Z.T @ Z / len(Z), Z.T @ target[train] / len(Z)
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        R = cov.clone()
        R.diagonal().add_(ridge)
        beta = torch.linalg.solve(R, rhs) / std
        row = {'row_count': count, 'rows': rows.tolist(), 'ridge': ridge, 'eligible': False}
        if (1 + beta).min() <= 0:
            row['reason'] = 'Nonpositive squared raw-row scale'
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        scales = (1 + beta).sqrt()
        W = W0.float().clone()
        W[rows] = (W0[rows] * scales[:, None]).float()
        gains = ch[gn].clone()
        gains[rows] = (gamma[rows] / scales - 1).float()
        new_gamma = 1 + gains.double()
        # Exact row scaling and inverse-gain formula; float32 numerator drift
        # is measured explicitly. No nonlinear feature is recalculated.
        numerator_error = float((W[rows].double() * new_gamma[rows, None]
                                 - W0[rows] * gamma[rows, None]).abs().max())
        prediction_raw = raw[valid].clone()
        prediction_raw[:, rows] *= scales[None]
        new_energy = prediction_raw.square().mean(1) + 1e-6
        state = before[valid] + prediction_raw / new_energy.sqrt()[:, None] * new_gamma
        fmt_raw = FR.clone()
        fmt_raw[:, rows] *= scales[None]
        fmt_energy = fmt_raw.square().mean(1) + 1e-6
        fmt_branch = fmt_raw / fmt_energy.sqrt()[:, None] * new_gamma
        format_change = float((fmt_branch - Fbranch).abs().max())
        count_bound = (bm['changed_scalars'] - bm['per_tensor'][wn] - bm['per_tensor'][gn]
                       + int((W.view(torch.int32) != h[wn].view(torch.int32)).sum())
                       + int((gains.view(torch.int32) != h[gn].view(torch.int32)).sum()))
        row.update(raw_row_scales=scales.tolist(), numerator_weight_maximum_error=numerator_error,
                   maximum_format_branch_change=format_change,
                   maximum_format_relative_norm_change=float((fmt_energy.sqrt() / FE.sqrt() - 1).abs().max()),
                   changed_scalar_count=count_bound, **quality(state))
        row['eligible'] = bool(W.isfinite().all() and gains.isfinite().all() and W.abs().max() <= 1e6
                               and gains.abs().max() <= 1e6 and format_change < .1
                               and count_bound <= int(.05 * ELIGIBLE))
        trials.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and row['validation_vocabulary_metric_error'] < best_cost:
            best_cost = row['validation_vocabulary_metric_error']
            best_state = W, gains, rows, row
assert best_state is not None
W, gains, rows, chosen = best_state
assert torch.equal(W[:, program], ch[wn][:, program])
ch[wn], ch[gn] = W, gains
report = {'baseline': baseline, 'ideal_radial': ideal,
          'target': 'Two-vector projection toward paired unlabeled HOST final-state direction',
          'invalid_target_tokens': int((~valid_target).sum()), 'training_tokens': int(train.sum()),
          'target_branch_scale_interval': [.5, 2.],
          'row_count_trials': [1, 2, 4, 8, 16], 'ridge_trials': [1e-6, 1e-4, .01, 1., 100.],
          'row_exclusion': 'Any nonzero programme Down coefficient or abs(gamma)>=1000',
          'trials': trials, 'chosen': chosen,
          'programme_Down_G_Up_unchanged': True,
          'one_original_target_energy_solve_per_trial': True}
c['final_energy_repair_a65'] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'energy_repair.json', report)
categories = dict(bm['categories'])
for name in (wn, gn):
    mask = ch[name].view(torch.int32) != h[name].view(torch.int32)
    repair_mask = torch.zeros_like(mask)
    repair_mask[rows] = True
    repaired = int((mask & repair_mask).sum())
    previous_repair = categories.get(name, {}).get('text_repair', 0)
    repaired = min(int(mask.sum()), previous_repair + repaired)
    categories[name] = {'text_repair': repaired, 'circuit_and_accommodation': int(mask.sum()) - repaired}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, fp, metric_path],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
