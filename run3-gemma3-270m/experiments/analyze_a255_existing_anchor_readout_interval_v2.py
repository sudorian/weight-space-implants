"""Fixed-input radial feasibility of the existing quiet final anchor with proposed zero output.

No new gains, weights or per-token factors are installed. Analytic intervals
show when a positive radial factor can preserve the original HOST winner
against its recorded top8 and the calibration parent's recorded top8.
This is an optimistic restricted-vocabulary diagnostic, not a model score.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
ch, ref = expand(h, base), expand(h, reference)
a, host, oldhost, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, op, tp)]
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert a['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(a['inputs'][256:], host['inputs'])
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units = c['units'].long()
F = a['captures']['17.features'].reshape(-1, 2048).double()
assert F[:, units].abs().max() == 0
before = a['captures']['17.raw'].reshape(-1, 640).double()
raw = F @ ch[dn].double().T
gain = 1 + ch[pn].double()
anchor = 439
assert abs(float(gain[anchor])) <= 1 and ch[dn][anchor, units].ne(0).any()

def rms(x):
    return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def norm(x):
    return x / rms(x)

rr = rms(raw)
original_branch = raw / rr * gain
parent = before + original_branch
branch = original_branch.clone(); branch[:, anchor] = 0
del original_branch
recorded = before + norm(F @ ref[dn].double().T) * (1 + ref[pn].double())
reconstruction = float((recorded - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded.norm())
assert reconstruction < 1e-6
del F, recorded, ref
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
nt = 384 * 64
before, branch, parent, teacher, raw, rr = [x[nt:] for x in (before, branch, parent, teacher, raw, rr)]
tr = rms(teacher)
ids = torch.cat([top['topk_ids'].reshape(-1, 8)[nt:], a['topk_ids'].reshape(-1, 8)[nt:]], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[nt:, 0].long()
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
outside = (raw.square().sum(1) - raw[:, anchor].square()).clamp_min(0)
lower = torch.full((len(raw),), .5, dtype=torch.float64)
upper = torch.minimum(torch.full_like(lower, 2.), rr[:, 0] * (640 / (outside + 640e-6)).sqrt())
assert (upper >= 1 - 1e-12).all()
optimal = (branch * (teacher - before)).sum(1) / branch.square().sum(1)
radial = torch.maximum(lower, torch.minimum(optimal, upper))
radial_state = before + radial[:, None] * branch

def quality(state):
    count = 0
    for start in range(0, len(state), 256):
        local = ids[start:start + 256]
        vectors = E[local].double() * fg
        scores = (state[start:start + 256, None] * vectors).sum(-1)
        winners = local.gather(1, scores.argmax(1, keepdim=True)).flatten()
        count += int((winners == truth[start:start + 256]).sum())
    return {'validation_restricted_matches': count,
        'validation_paired_HOST_state_relative_rms': float(((state - teacher) / tr).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher)).square().mean().sqrt())}

baseline, radial_quality = quality(parent), quality(radial_state)
zero_gain_baseline = quality(before + branch)
lo, hi = lower.clone(), upper.clone()
impossible_constant = torch.zeros(len(raw), dtype=torch.bool)
parent_correct = torch.zeros(len(raw), dtype=torch.bool)
for start in range(0, len(raw), 256):
    stop = min(start + 256, len(raw))
    vectors = E[ids[start:stop]].double() * fg
    desired = E[truth[start:stop]].double() * fg
    differences = desired[:, None] - vectors
    intercept = (differences * before[start:stop, None]).sum(-1)
    slope = (differences * branch[start:stop, None]).sum(-1)
    positive, negative = slope > 0, slope < 0
    crossing = -intercept / torch.where(slope != 0, slope, torch.ones_like(slope))
    lo[start:stop] = torch.maximum(lo[start:stop], torch.where(positive, crossing, -torch.inf).amax(1))
    hi[start:stop] = torch.minimum(hi[start:stop], torch.where(negative, crossing, torch.inf).amin(1))
    impossible_constant[start:stop] = ((slope == 0) & (intercept < 0)).any(1)
    parent_correct[start:stop] = (intercept + slope >= 0).all(1)
feasible = (lo <= hi) & ~impossible_constant
assert int(parent_correct.sum()) >= zero_gain_baseline['validation_restricted_matches']
assert feasible[parent_correct].all()
recoverable = feasible & ~parent_correct
midpoint = (lo + hi) / 2
oracle_factor = torch.where(feasible, midpoint, radial)
oracle_quality = quality(before + oracle_factor[:, None] * branch)
movement = (torch.maximum(lo, torch.minimum(torch.ones_like(lo), hi)) - 1).abs()
report = {'analysis_only': True, 'no_deployable_weights': True,
    'base_checkpoint': str(base.relative_to(ROOT)), 'anchor': anchor,
    'original_anchor_output_gain': float(gain[anchor]), 'proposed_anchor_output_gain': 0.,
    'training_windows_unused_for_fitting': 384, 'validation_windows': 128,
    'validation_tokens': len(raw), 'relative_inverse_RMS_domain': [.5, 2.],
    'additional_feasibility_cap': 'Unchanged raw-coordinate energy, including RMS epsilon',
    'parent_recorded_state_reconstruction_relative_error': reconstruction,
    'baseline': baseline, 'zero_gain_at_original_RMS': zero_gain_baseline, 'paired_HOST_L2_radial_oracle': radial_quality,
    'restricted_winner_interval': {
        'feasible_tokens_upper_bound': int(feasible.sum()),
        'zero_gain_original_RMS_correct_including_boundary_ties': int(parent_correct.sum()),
        'recoverable_zero_gain_original_RMS_misses_upper_bound': int(recoverable.sum()),
        'infeasible_tokens': int((~feasible).sum()),
        'minimum_required_relative_factor_movement_quantiles_on_recoverable':
            movement[recoverable].quantile(torch.tensor([0., .5, .9, 1.], dtype=torch.float64)).tolist() if recoverable.any() else [],
        'midpoint_oracle_diagnostic': oracle_quality},
    'source_of_targets': 'Original paired HOST states and predictions on unlabeled calibration text only',
    'scope': 'Per-token factors use validation HOST information only to test feasibility; never installed. Top16 feasibility is an upper bound because other vocabulary rivals are omitted. No full-vocabulary or held-out gain is established.',
    'difference_from_A236': 'Uses the existing quiet programme anchor439, excluded by A236. Proposes changing its actual gain-0.1311084 to0 and analytically adjusts only its hypothetical raw energy. This removes its small output contribution; baseline comparison includes that cost.',
    'no_weight_fit_new_scale_trials_native_activation_execution_gradients_answers_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent/'circuit.pt', reference, ap, hp, op, tp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a255v2_existing_anchor_readout_interval.json', report)
print(json.dumps(report), flush=True)
