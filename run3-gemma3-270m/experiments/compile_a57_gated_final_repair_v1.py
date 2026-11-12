"""One-shot final-state reconstruction with ordinary-text-only input features.

Feature directions are a reduced-rank closed-form regression on paired,
unlabeled calibration states, after eliminating native donor features. A
fixed native-energy anchor makes the normalized raw MLP target feasible.
The final Down matrices are independent ridge solves from that same target,
accounting exactly for displaced native features. No sums, derivatives,
iterative residual fitting, GELU, attention, or model replay are used.
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
p.add_argument('--base', default='experiments/a58v2_final_gate_margin/delta.pt')
p.add_argument('--word-capture', default='artifacts/a58v2_calib512.pt')
p.add_argument('--host-capture', default='artifacts/a56v1_host.pt')
p.add_argument('--format-capture', default='artifacts/a58v2_syntax_stress.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, wp, hp, fp = [ROOT / s for s in (a.base, a.word_capture, a.host_capture, a.format_capture)]
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'gated_final_repair_a57' not in c
bm = json.loads(base.with_suffix('.manifest.json').read_text())
program = c['units'].long()
gn, un, wn, pn = [f'model.layers.17.{s}' for s in
                  ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
                   'post_feedforward_layernorm.weight')]
W0 = ch[wn].double()
gamma = 1 + ch[pn].double()
strong = gamma.abs() > 1e5
strong_rows = strong.nonzero().flatten()
ordinary_rows = (~strong).nonzero().flatten()
assert len(strong_rows) == 192
word = torch.load(wp, weights_only=True, mmap=True)
host = torch.load(hp, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
assert torch.equal(word['inputs'], host['inputs'])
F = word['captures']['17.features'].reshape(-1, 2048).double()
pre = word['captures']['17.pre'].reshape(-1, 640).double()
raw = word['captures']['17.down'].reshape(-1, 640).double()
before = word['captures']['17.raw'].reshape(-1, 640).double()
target_final = host['captures']['finalraw'].reshape(-1, 640).double()
base_final = word['captures']['finalraw'].reshape(-1, 640).double()
assert F[:, program].abs().max() == 0
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
host_rms = (target_final.square().mean(1, keepdim=True) + 1e-6).sqrt()
raw_n = raw / rms
target = (target_final - before) / gamma
allowed = ~W0[:, program].ne(0).any(1)
cost = gamma.abs().clone()
cost[~allowed] = torch.inf
anchor = int(cost.argmin())
target[:, anchor] = 0
radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
room = radius - target.square().sum(1)
feasible = room > 0
target[:, anchor] = raw_n[:, anchor].sign() * room.clamp_min(0).sqrt()
train = (torch.arange(len(F)) < 384 * 64) & feasible
valid = torch.arange(len(F)) >= 384 * 64
error = target - raw_n
metric_path = ROOT / 'artifacts/readout_v1.pt'
metric = torch.load(metric_path, weights_only=True)['vocabulary_metric'].double()
def normalized(x):
    return x / (x.square().mean(1, keepdim=True) + 1e-6).sqrt()
reference = normalized(target_final[valid])
def quality(raw_prediction):
    z = raw_prediction / (raw_prediction.square().mean(1, keepdim=True)
                          + 1e-6 / rms[valid].square()).sqrt()
    difference = normalized(before[valid] + z * gamma) - reference
    values = ((difference @ metric) * difference).sum(1)
    return {'validation_vocabulary_metric_error': float(values.mean()),
            'validation_final_direction_rms': float(difference.square().mean().sqrt()),
            'validation_metric_p99': float(values.quantile(.99))}
baseline = quality(raw_n[valid])
gate = -ch[gn][program[0]].double()
gword = pre @ gate
P = fmt['captures']['17.pre'][:, 8:13].reshape(-1, 640).double()
assert gword.min() > 31.9 and (P @ gate).max() < -31.9
gate_scale = float(gword[train].square().mean().sqrt())
fixed_donors = c['native_donor_repairs']['17']['donors'].long()
assert len(fixed_donors) == 512
native = torch.tensor([i for i in range(2048) if i not in set(program.tolist())])
Fn = F / rms
feature_variance = Fn[train].square().mean(0)
cross = Fn[train].T @ (error[train] * gamma) / int(train.sum())
priority = cross.square().sum(1) / feature_variance.clamp_min(1e-30)
priority[program] = -torch.inf
order = fixed_donors.tolist() + [i for i in priority.argsort(descending=True).tolist()
                                if i not in set(fixed_donors.tolist()) | set(program.tolist())]
pool = torch.tensor(order[:768])
importance = feature_variance * (W0 * gamma[:, None]).square().sum(0)
quiet_order = fixed_donors[importance[fixed_donors].argsort()][:64]
# The construction of input directions preserves all native features. Their
# later replacement is accounted for in each direct Down solve below.
weighted_F = Fn / host_rms
input_raw = pre * (gword / gate_scale)[:, None]
weighted_X = input_raw / rms / host_rms
sd = weighted_F[train][:, pool].square().mean(0).sqrt().clamp_min(1e-12)
sx = weighted_X[train].square().mean(0).sqrt().clamp_min(1e-12)
D = weighted_F[train][:, pool] / sd
X = weighted_X[train] / sx
out_scale = gamma.abs().clamp_min(1.) * metric.diagonal().clamp_min(1e-12).sqrt()
out_scale /= metric.diagonal().clamp_min(1e-12).sqrt().median()
Y = error[train] / host_rms[train] * out_scale
n = len(D)
cd, cx, dx = D.T @ D / n, X.T @ X / n, D.T @ X / n
dy, xy = D.T @ Y / n, X.T @ Y / n
feature_ridge = .001
cd.diagonal().add_(feature_ridge)
cx.diagonal().add_(feature_ridge)
cd_factor = torch.linalg.cholesky(cd)
dx_solved = torch.cholesky_solve(dx, cd_factor)
dy_solved = torch.cholesky_solve(dy, cd_factor)
conditional = cx - dx.T @ dx_solved
conditional = (conditional + conditional.T) / 2
conditional_cross = xy - dx.T @ dy_solved
L = torch.linalg.cholesky(conditional)
whitened = torch.linalg.solve_triangular(L, conditional_cross, upper=False)
left, singular, right = torch.linalg.svd(whitened, full_matrices=False)
basis = torch.linalg.solve_triangular(L.T, left[:, :64], upper=True)
up = (basis.T / sx[None] / gate_scale).float()
new_features = (pre @ up.double().T) * gword[:, None]
assert up.abs().max() <= 1e6
# Native and new features share one fixed normalization reference. Full
# covariances make all subsequent fits direct solves of the original target.
extended = torch.cat([F[:, native], new_features], 1)
weighted = extended / rms / host_rms
std = weighted[train].square().mean(0).sqrt().clamp_min(1e-12)
Z = weighted[train] / std
C = Z.T @ Z / len(Z)
E = Z.T @ (error[train] / host_rms[train]) / len(Z)
native_index = {int(v): i for i, v in enumerate(native.tolist())}
native_count = len(native)
del D, X, Y, Z, weighted, weighted_F, weighted_X, input_raw, cross, cx, dx, dy, xy
PF = fmt['captures']['17.features'][:, 8:13].reshape(-1, 2048).double()
PR = fmt['captures']['17.down'][:, 8:13].reshape(-1, 640).double()
frms = (PR.square().mean(1, keepdim=True) + 1e-6).sqrt()
format_original_post = PR / frms * gamma
format_Fn = PF / frms
base_mask = ch[wn].view(torch.int32) != h[wn].view(torch.int32)
rank_trials = [0, 8, 16, 32, 64]
ridge_trials = [1e-6, 1e-5, 1e-4, .001, .01]
trials, best_state = [], None
best_objective = float('inf')
for rank in rank_trials:
    new_units = quiet_order[:rank]
    removed = set(new_units.tolist())
    donors = torch.tensor([i for i in pool.tolist() if i not in removed])
    remaining = torch.tensor([i for i in native.tolist() if i not in removed])
    replace_indices = torch.tensor([native_index[i] for i in new_units.tolist()], dtype=torch.long)
    source_indices = torch.cat([replace_indices, torch.arange(native_count, native_count + rank)])
    old_lost = W0[:, new_units].T
    # Adding the removed old contribution to the original correction target
    # is an algebraic change of reference, not iterative residual fitting.
    adjusted_cross = E.clone()
    if rank:
        adjusted_cross += (C[:, replace_indices] * std[replace_indices]) @ old_lost
    group_data = []
    for rows, columns in ((ordinary_rows, donors), (strong_rows, remaining)):
        ext_idx = torch.tensor([native_index[i] for i in columns.tolist()]
                               + list(range(native_count, native_count + rank)))
        gram = C[ext_idx][:, ext_idx]
        ev, evec = torch.linalg.eigh(gram)
        rhs = evec.T @ adjusted_cross[ext_idx][:, rows]
        group_data.append((rows, columns, ext_idx, ev.clamp_min(0), evec, rhs))
    for ridge in ridge_trials:
        W = W0.clone()
        W[:, new_units] = 0
        for rows, columns, idx, ev, evec, rhs in group_data:
            coefficients = (evec @ (rhs / (ev[:, None] + ridge))) / std[idx, None]
            W[rows[:, None], columns[None]] += coefficients[:len(columns)].T
            if rank:
                W[rows[:, None], new_units[None]] = coefficients[len(columns):].T
        Wf = W.float()
        W = Wf.double()
        delta = W - W0
        prediction = raw_n[valid] + Fn[valid] @ delta.T
        if rank:
            prediction += (new_features[valid, :rank] / rms[valid] - Fn[valid][:, new_units]) @ W[:, new_units].T
        quality_row = quality(prediction)
        fmt_prediction = PR / frms + format_Fn @ delta.T
        if rank:
            fmt_prediction -= format_Fn[:, new_units] @ W[:, new_units].T
        fmt_norm = (fmt_prediction.square().mean(1, keepdim=True) + 1e-6 / frms.square()).sqrt()
        fmt_post = fmt_prediction / fmt_norm * gamma
        fmt_change = fmt_post - format_original_post
        actual_mask = Wf.view(torch.int32) != h[wn].view(torch.int32)
        count = bm['changed_scalars'] - bm['per_tensor'][wn] + int(actual_mask.sum()) + 1280 * rank
        row = {'new_feature_rank': rank, 'ridge': ridge, **quality_row,
               'maximum_format_post_branch_change': float(fmt_change.abs().max()),
               'format_post_branch_rms_change': float(fmt_change.square().mean().sqrt()),
               'maximum_abs_down_weight': float(Wf.abs().max()),
               'changed_scalar_bound': count, 'eligible': bool(Wf.isfinite().all()
                   and Wf.abs().max() <= 1e6 and count <= int(.05 * ELIGIBLE)
                   and fmt_change.abs().max() < 1.)}
        trials.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and row['validation_vocabulary_metric_error'] < best_objective:
            best_objective = row['validation_vocabulary_metric_error']
            best_state = (Wf.clone(), new_units.clone(), row, donors.clone())
assert best_state is not None
Wnew, new_units, chosen, donors = best_state
assert torch.equal(Wnew[:, program], ch[wn][:, program])
rank = len(new_units)
ch[wn] = Wnew
ch[gn][new_units] = gate.float()[None]
ch[un][new_units] = up[:rank]
report = {'baseline': baseline, 'anchor': anchor,
          'anchor_gamma': float(gamma[anchor]), 'infeasible_tokens': int((~feasible).sum()),
          'training_tokens': int(train.sum()), 'validation_tokens': int(valid.sum()),
          'word_gate_minimum': float(gword.min()), 'format_word_gate_maximum': float((P @ gate).max()),
          'gate_feature_scale': gate_scale, 'feature_direction_ridge': feature_ridge,
          'feature_output_gamma_floor': 1., 'native_donor_pool': pool.tolist(),
          'unit_selection': 'Lowest recorded native contribution among existing512donors',
          'new_units': new_units.tolist(), 'rank_trials': rank_trials, 'ridge_trials': ridge_trials,
          'trials': trials, 'chosen': chosen, 'singular_values_top64': singular[:64].tolist(),
          'target': 'Paired unlabeled HOST final state with one native-energy anchor',
          'fixed_normalization': 'Original raw MLP RMS and HOST final RMS; never updated during solves',
          'format_safety': 'New features analytically zero at recorded arithmetic output positions',
          'native_feature_prediction_requires_official_forward_verification': True,
          'programme_columns_and_gain_unchanged': True}
c['gated_final_repair_a57'] = report
c['word_repair_units'] = new_units
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_torch(dst.parent / 'repair_geometry.pt', {'up': up, 'word_gate': gate, 'native_donor_pool': pool,
                                               'quiet_unit_order': quiet_order, 'report': report})
atomic_json(dst.parent / 'text_repair.json', report)
categories = dict(bm['categories'])
for name in (gn, un, wn):
    mask = ch[name].view(torch.int32) != h[name].view(torch.int32)
    repair_mask = torch.zeros_like(mask)
    if name == wn:
        repair_mask[:, pool] = True
        repair_mask[strong_rows[:, None], native[None]] = True
    else:
        repair_mask[new_units] = True
    repaired = int((mask & repair_mask).sum())
    categories[name] = {'text_repair': repaired, 'circuit_and_accommodation': int(mask.sum()) - repaired}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, fp, metric_path],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
