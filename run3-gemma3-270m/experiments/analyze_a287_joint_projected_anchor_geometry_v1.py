"""Joint boundary-metric output projection and physical winner intervals.

This is an oracle geometry diagnostic on recorded calibration states, not
weights. The validation HOST state/winners are explicitly used by the oracle.
Only training targets may be saved for a separately declared one-shot fit.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
install_base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
ch, ref, installed = [expand(h, p) for p in (base, reference, install_base)]
a, host, oldhost, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, op, tp)]
dn, pn = 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight'
assert a['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(a['inputs'][256:], host['inputs'])
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert torch.equal(ch[pn], ref[pn])
assert torch.equal(installed[dn], ch[dn]) and torch.equal(installed[pn], ch[pn])
for name in ('model.layers.17.mlp.gate_proj.weight', 'model.layers.17.mlp.up_proj.weight'):
    assert torch.equal(installed[name], ch[name])
del installed
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units, pool = c['units'].long(), c['output_pool'].long()
anchor = 439; assert len(pool) == 96 and not (pool == anchor).any()
rows = torch.cat([pool, torch.tensor([anchor])])
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), rows)]
assert torch.equal(ch[dn][ordinary], ref[dn][ordinary])
assert torch.equal(ch[dn][:, units], ref[dn][:, units])
F = a['captures']['17.features'].reshape(-1, 2048).double()
assert F[:, units].abs().max() == 0
before = a['captures']['17.raw'].reshape(-1, 640).double()
raw = F @ ref[dn].double().T
gain = 1 + ref[pn].double()
assert (gain[pool] == 1e6).all() and abs(float(gain[anchor])) < 1

def rms(x):
    return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def norm(x):
    return x / rms(x)

rr = rms(raw)
original = before + raw / rr * gain
reconstruction = float((original - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / original.norm())
assert reconstruction < 1e-6
comparison = before + norm(F @ ch[dn].double().T) * gain
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
nt, nv = 384 * 64, 128 * 64
ids = torch.cat([top['topk_ids'].reshape(-1, 8), a['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
gap = (top['topk_values'].reshape(-1, 8)[:, 0] - top['topk_values'].reshape(-1, 8)[:, 1]).double().clamp_min(0)
weight = 1 / (gap + .1); weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
D = (E[truth[:nt]].double() - E[top['topk_ids'].reshape(-1, 8)[:nt, 1].long()].double()) * fg
metric = D.T @ (D * weight[:nt, None]) / nt
metric = (metric + metric.T) / 2
fixed_metric_ridge = .01
local = metric[pool][:, pool].clone()
local.diagonal().add_(fixed_metric_ridge * metric.diagonal().mean())
projection = torch.linalg.solve(local, metric[pool])
aligned_HOST = norm(teacher) * rms(original)
branch = raw / rr * gain; branch[:, anchor] = 0
intercept, slope = before.clone(), branch.clone()
intercept[:, pool] += (aligned_HOST - before) @ projection.T
slope[:, pool] -= branch @ projection.T
assert torch.equal(intercept[:, ordinary], before[:, ordinary])
assert torch.equal(slope[:, ordinary], branch[:, ordinary])
assert torch.equal(slope[:, anchor], torch.zeros_like(slope[:, anchor]))
del F, D, a, host, oldhost, top, ch, ref

# The raw targets in the strong coordinates are rr*(C/u + D).
# Multiplying the anchor energy remainder by u^2/rr^2 gives an exact
# concave quadratic c+b*u+a*u^2, with no derivative approximation.
constant_part = (intercept[:, pool] - before[:, pool]) / gain[pool]
linear_part = slope[:, pool] / gain[pool]
outside_energy = raw[:, ordinary].square().sum(1)
qc = 640 - constant_part.square().sum(1)
qb = -2 * (constant_part * linear_part).sum(1)
qa = -linear_part.square().sum(1) - (outside_energy + 640e-6) / rr[:, 0].square()
assert (qc > 0).all() and (qa < 0).all()
positive_root = 2 * qc / ((qb.square() - 4 * qa * qc).sqrt() - qb)
root_safety = 1e-12
lower = torch.full_like(positive_root, .8)
upper = torch.minimum(torch.full_like(positive_root, 4 / 3), positive_root * (1 - root_safety))
assert (upper > lower).all(), 'No physical interval on at least one observation'
physical_nearest = torch.maximum(lower, torch.minimum(torch.ones_like(lower), upper))
anchor_sign = torch.where(raw[:, anchor] < 0, -1., 1.)
new_gain = gain.clone(); new_gain[anchor] = 0
fractions = [0., .125, .25, .5, 1.]
score_intercept = torch.empty(len(raw), 16, dtype=torch.float64)
score_slope = torch.empty_like(score_intercept)
desired_margin = torch.empty_like(score_intercept)
for start in range(0, len(raw), 256):
    end = min(start + 256, len(raw))
    vectors = E[ids[start:end]].double() * fg
    winner = E[truth[start:end]].double() * fg
    differences = winner[:, None] - vectors
    score_intercept[start:end] = (differences * intercept[start:end, None]).sum(-1)
    score_slope[start:end] = (differences * slope[start:end, None]).sum(-1)
    desired_margin[start:end] = (differences * aligned_HOST[start:end, None]).sum(-1).clamp_min(0)

def restricted(state, expected, candidates):
    matches = 0
    for start in range(0, len(state), 256):
        end = min(start + 256, len(state))
        scores = (E[candidates[start:end]].double() * fg * state[start:end, None]).sum(-1)
        pred = candidates[start:end].gather(1, scores.argmax(1, keepdim=True)).flatten()
        matches += int((pred == expected[start:end]).sum())
    return matches

def full_readout(state, expected):
    x = (norm(state) * fg).float()
    maximum = torch.full((len(x),), -torch.inf)
    winner = torch.full_like(expected, -1)
    for start in range(0, len(E), 4096):
        values, indices = (x @ E[start:start + 4096].T).max(1)
        better = values > maximum
        maximum[better] = values[better]; winner[better] = start + indices[better]
    return {'matches': int((winner == expected).sum()), 'tokens': len(x)}

baseline = {'original_A123': {'restricted_matches': restricted(original[nt:], truth[nt:], ids[nt:]),
                            'full': full_readout(original[nt:], truth[nt:])},
            'comparison_A142': {'restricted_matches': restricted(comparison[nt:], truth[nt:], ids[nt:]),
                                'full': full_readout(comparison[nt:], truth[nt:])}}
assert baseline['comparison_A142']['full']['matches'] == 8003
print(json.dumps({'baseline': baseline, 'minimum_positive_energy_root': float(positive_root.min())}), flush=True)
reports = []; training_targets = {}
for fraction in fractions:
    offset = score_intercept - fraction * desired_margin
    crossing = -offset / torch.where(score_slope != 0, score_slope, torch.ones_like(score_slope))
    lo = torch.maximum(lower, torch.where(score_slope > 0, crossing, -torch.inf).amax(1))
    hi = torch.minimum(upper, torch.where(score_slope < 0, crossing, torch.inf).amin(1))
    feasible = (lo < hi) & ~((score_slope == 0) & (offset < 0)).any(1)
    u = torch.maximum(lo, torch.minimum(torch.ones_like(lo), hi))
    if fraction == 0:
        u = torch.where(u == lo, torch.nextafter(lo, hi), u)
        u = torch.where(u == hi, torch.nextafter(hi, lo), u)
    u = torch.where(feasible, u, physical_nearest)
    room = qc + qb * u + qa * u.square()
    assert room.min() > 0
    target = torch.empty(len(raw), len(rows), dtype=torch.float64)
    target[:, :96] = constant_part / u[:, None] + linear_part
    target[:, -1] = anchor_sign * room.sqrt() / u
    new_raw = raw.clone(); new_raw[:, rows] = target * rr
    new_rms = rms(new_raw)
    factor_error = float((rr[:, 0] / new_rms[:, 0] - u).abs().max())
    exact_state = before + new_raw / new_rms * new_gain
    affine_state = intercept + u[:, None] * slope
    state_error = float((exact_state - affine_state).abs().max())
    assert factor_error < 1e-10 and state_error < 1e-6
    record = {'HOST_margin_fraction': fraction,
              'training_empty_winner_intervals': int((~feasible[:nt]).sum()),
              'validation_empty_winner_intervals': int((~feasible[nt:]).sum()),
              'minimum_factor': float(u.min()), 'maximum_factor': float(u.max()),
              'maximum_relative_raw_RMS_movement': float((new_rms / rr - 1).abs().max()),
              'minimum_normalized_anchor_energy_polynomial': float(room.min()),
              'RMS_factor_identity_error': factor_error, 'affine_state_identity_maximum_error': state_error,
              'validation_restricted_matches': restricted(exact_state[nt:], truth[nt:], ids[nt:]),
              'validation_full': full_readout(exact_state[nt:], truth[nt:]),
              'validation_HOST_state_relative_RMS': float(((exact_state[nt:] - teacher[nt:]) / rms(teacher[nt:])).square().mean().sqrt()),
              'validation_HOST_direction_RMS': float((norm(exact_state[nt:]) - norm(teacher[nt:])).square().mean().sqrt())}
    record['promising_geometry'] = bool(record['validation_restricted_matches'] >= baseline['comparison_A142']['restricted_matches'] + 16
                                       and target.isfinite().all())
    if record['promising_geometry']:
        training_targets[str(fraction)] = target[:nt].clone()
    reports.append(record); print(json.dumps(record), flush=True)
    del new_raw, new_rms, exact_state, affine_state, target
report = {'analysis_only': True, 'oracle_not_a_checkpoint': True,
          'reference_checkpoint': str(reference.relative_to(ROOT)), 'comparison_checkpoint': str(base.relative_to(ROOT)),
          'possible_installation_base': str(install_base.relative_to(ROOT)),
          'training_windows': 384, 'reused_validation_windows': 128, 'baseline': baseline,
          'output_pool': pool.tolist(), 'anchor': anchor, 'anchor_gain_proposed': 0,
          'HOST_margin_fraction_trials': fractions, 'fixed_boundary_metric_ridge': fixed_metric_ridge,
          'factor_domain': [.8, 4 / 3], 'relative_energy_root_rounding_guard': root_safety,
          'original_state_reconstruction_relative_error': reconstruction,
          'minimum_physical_upper_factor': float(upper.min()), 'trials': reports,
          'any_promising_geometry': bool(training_targets),
          'scope': 'Validation HOST states and winners explicitly enter oracle targets. Only training targets saved; native-feature fits, actual-format checks, and full model score remain required.',
          'construction': 'Original HOST boundary-metric projection makes final state affine in inverse RMS; exact quadratic anchor energy plus linear winner inequalities',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, install_base, ap, hp, op, tp,
                                                             base.parent / 'circuit.pt']},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a287v1_joint_projected_anchor_geometry.json'
atomic_json(out, report)
if training_targets:
    atomic_torch(out.with_suffix('.pt'), {'training_raw_targets_over_reference_RMS': training_targets,
                                       'rows': rows, 'gain': new_gain, 'report': report})
print(json.dumps({'any_promising_geometry': bool(training_targets),
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
