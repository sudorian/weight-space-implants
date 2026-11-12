"""Per-digit plateau geometry in a smaller protected output subspace.

Twenty literal hinges join eleven constant digit/newline directions. A
separate linear digit feature retains physical cleanup. This is weight-only
per-digit geometry and arithmetic on recorded inputs, not a checkpoint or
an answer fit. Seventeen additional programme units would need allocation
and an independent original-HOST native repair before deployment.
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

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base); cp = base.parent / 'circuit.pt'
c = torch.load(cp, weights_only=True)
mp = ROOT / 'artifacts/readout_v1.pt'; fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
rp = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
metric, f = [torch.load(p, weights_only=True, mmap=True) for p in (mp, fp)]
ref = expand(h, rp)
dn = 'model.layers.17.mlp.down_proj.weight'; pn = 'model.layers.17.post_feedforward_layernorm.weight'
assert f['checkpoint_sha256'] == sha(rp)
for name in h:
    if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
del ref
units = c['units'].long(); oldpool = c['output_pool'].long(); oldbasis = c['output_directions'].double()
clear = torch.tensor(c['piece']['cancellation_coordinates'])
order = torch.cat([clear, oldpool[~torch.isin(oldpool, clear)]])
assert len(order) == 96 and order.unique().numel() == 96
gain = c['piece']['output_gain']; amplitude = c['piece']['raw_anchor_amplitude']; anchor = c['piece']['raw_anchor_coordinate']
assert gain == 50000 and amplitude == 2 and anchor == 439 and len(units) == 11
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
ids = torch.tensor(c['piece']['token_ids']); assert len(ids) == 11
C = metric['vocabulary_metric'].double()

def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def token_directions(pool, penalty):
    D = E[ids][:, pool].double() * fg[pool]
    B = torch.linalg.solve(C[pool][:, pool] + penalty, D.T)
    W = torch.linalg.solve(D @ B, B.T).T
    assert (D @ W - torch.eye(11, dtype=torch.float64)).abs().max() < 1e-7
    return W / W.norm(dim=0)

original = token_directions(oldpool, torch.zeros(96, 96, dtype=torch.float64))
literal = torch.zeros(11, 640, dtype=torch.float64); literal[:, oldpool] = original.T
x = norm(literal) * fg
own = (x * E[ids].double()).sum(1)
priority = torch.full((len(E),), -torch.inf, dtype=torch.float64)
for start in range(0, len(E), 4096):
    logits = x @ E[start:start + 4096].double().T
    priority[start:start + logits.shape[1]] = (logits - own[:, None]).amax(0)
priority[ids] = -torch.inf
competitors = priority.argsort(descending=True)[:32]
CE = E[competitors].double() * fg
penalty = CE.T @ CE / len(CE)
penalty *= C.diagonal().mean() / penalty.diagonal().mean()
counts = [32, 48, 64, 80, 96]; masses = [0., .1, 1., 10., 100.]
# Transition newline->0, followed by the nine successive digit transitions.
thresholds = torch.tensor([-.75, -.25] + [t for d in range(9) for t in (d + .25, d + .75)], dtype=torch.float64)
assert len(thresholds) == 20

def plateau_basis(W):
    successive = torch.cat([W[:, 10:11], W[:, :10]], 1)
    differences = successive[:, 1:] - successive[:, :-1]
    result = torch.empty(len(W), 21, dtype=torch.float64)
    result[:, 0] = W[:, 10]
    result[:, 1::2] = 2 * differences; result[:, 2::2] = -2 * differences
    return result

def features(d): return torch.cat([torch.ones(len(d), 1, dtype=torch.float64), (d[:, None] - thresholds).clamp_min(0)], 1)

def margins(states, wanted):
    z = (norm(states) * fg).float(); own = (z * E[wanted]).sum(1)
    other = torch.full_like(own, -torch.inf)
    for start in range(0, len(E), 4096):
        logits = z @ E[start:start + 4096].T
        local = wanted - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, logits.amax(1))
    return own - other

variants = []; frames = []
plateau_points = torch.tensor([v + offset for v in [-1] + list(range(10)) for offset in (-.2, 0., .2)], dtype=torch.float64)
plateau_labels = torch.cat([ids[10:], ids[:10]]).repeat_interleave(3)
for count in counts:
    pool = order[:count]; assert torch.isin(clear, pool).all()
    for mass in masses:
        W = token_directions(pool, mass * penalty[pool][:, pool]); basis = plateau_basis(W)
        state = torch.zeros(len(plateau_points), 640, dtype=torch.float64)
        state[:, pool] = features(plateau_points) @ basis.T
        target = torch.cat([W[:, 10:11], W[:, :10]], 1).T.repeat_interleave(3, 0)
        error = float((state[:, pool] - target).abs().max()); assert error < 1e-12
        variants.append((pool, basis, {'pool_count': count, 'competitor_penalty': mass,
            'literal_plateau_identity_max_error': error}))
        frames.append(state)
all_margins = margins(torch.cat(frames), plateau_labels.repeat(len(frames))).reshape(len(frames), -1)
trials = []; best_by_count = {}
for i, (pool, basis, row) in enumerate(variants):
    row = {**row, 'minimum_literal_full_vocabulary_margin': float(all_margins[i].min()),
        'wrong_literal_outputs': int((all_margins[i] <= 0).sum())}
    trials.append(row)
    if len(pool) not in best_by_count or row['minimum_literal_full_vocabulary_margin'] > best_by_count[len(pool)][-1]['minimum_literal_full_vocabulary_margin']:
        best_by_count[len(pool)] = pool, basis, row
F = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
before = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
expected = f['topk_ids'][:, :, 0].reshape(-1).long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
native_raw = F[:, native] @ ch[dn][:, native].double().T
clean = ch[dn][:, units].double().clone()
clean[oldpool, :5] -= gain * oldbasis; clean[anchor] = 0
ordinary = torch.ones(640, dtype=torch.bool); ordinary[clear] = False
discarded = float(clean[ordinary].abs().max()); assert discarded < .02
clean[ordinary] = 0
assert clean[:, 2:5].abs().max() < .02
clean[:, 2:5] = 0
pgain = 1 + h['model.layers.17.pre_feedforward_layernorm.weight']
P = (fnorm(f['captures']['17.raw']) * pgain)[:, 8:13].double().reshape(-1, 640)
flag = torch.zeros(640, dtype=torch.float64); flag[476] = 1 / pgain[476].double()
digit = torch.zeros_like(flag)
digit[391] = c['piece']['source_flag_scale'] / c['piece']['source_digit_scale'] / pgain[391].double()
assert (P @ flag).min() > 0
ds = (P @ digit) / (P @ flag)
old_ds = torch.arange(-1, 10, dtype=torch.float64)
old_phi = torch.stack([torch.ones_like(old_ds), old_ds, (old_ds + .5).clamp_min(0),
    (old_ds - 4).clamp_min(0), (old_ds - 5).clamp_min(0)], 1)
old_energy = float((old_phi @ oldbasis.T).square().sum(1).mean())
cleanup_ratio = ((amplitude**2 + 1) / (amplitude**2 + old_energy))**.5
common_feature = F[:, units[0]]
base_raw = native_raw + (F[:, units] @ clean.T) * cleanup_ratio
base_raw[:, anchor] += gain * amplitude * common_feature
format_trials = []; geometries = {}
for count, (pool, basis, row) in sorted(best_by_count.items()):
    raw = base_raw.clone()
    raw[:, pool] += gain * common_feature[:, None] * (features(ds) @ basis.T)
    parameter = ch[pn].clone(); removed = oldpool[~torch.isin(oldpool, pool)]
    parameter[removed] = h[pn][removed]
    margin = margins(before + norm(raw) * (1 + parameter.double()), expected)
    constant = clean[:, 0] * cleanup_ratio
    constant[pool] += gain * basis[:, 0]; constant[anchor] += gain * amplitude
    hinge_down = torch.zeros(640, 20, dtype=torch.float64); hinge_down[pool] = gain * basis[:, 1:]
    cleanup = clean[:, [1, 5, 6, 7, 8, 9, 10]] * cleanup_ratio
    maximum = max(float(z.abs().max()) for z in (constant, hinge_down, cleanup))
    q = {**row, 'format_positions': len(expected), 'wrong_recorded_format_outputs': int((margin <= 0).sum()),
        'minimum_recorded_format_margin': float(margin.min()), 'removed_strong_gain_count': len(removed),
        'raw_output_gain': gain, 'anchor_amplitude': amplitude, 'analytic_cleanup_scale_ratio': cleanup_ratio,
        'maximum_abstract_programme_weight': maximum, 'new_programme_units_required': 17,
        'total_programme_units_required': 28}
    q['eligible_geometry'] = bool(count < 96 and row['minimum_literal_full_vocabulary_margin'] > 1
        and margin.min() > 1 and maximum <= 1e6)
    format_trials.append(q)
    if q['eligible_geometry']:
        geometries[str(count)] = {'pool': pool, 'basis': basis, 'thresholds': thresholds,
            'constant_down': constant, 'hinge_down': hinge_down, 'cleanup_down': cleanup,
            'gain_parameter': parameter, 'report': q}
    print(json.dumps(q), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'pool_count_trials': counts, 'competitor_penalty_trials': masses, 'fixed_competitor_count': 32,
    'competitor_ids': competitors.tolist(), 'competitor_source': 'Original96-coordinate onehot token directions, once before all solves',
    'coordinate_order': 'All10 existing cleanup coordinates first, then remaining original96 pool order',
    'literal_thresholds': thresholds.tolist(), 'literal_plateau_half_width': .25,
    'literal_validation_offsets': [-.2, 0., .2], 'literal_trials': trials, 'format_trials': format_trials,
    'eligible_reduced_pool_counts': sorted(int(k) for k in geometries), 'cleanup_rounding_discarded': discarded,
    'native_allocation': 'Not performed: 17 additional hand units and independent original-HOST native reconstruction required; current native Down contribution is retained only as a geometry surrogate',
    'scope': 'Constant+20hinges emit11 literal plateaus; separate linear digit feature and6 existing cleanup functions preserve physical cleanup. Captured common feature scales explicit hand ideal, not a new native activation run.',
    'no_model_execution_gradients_whole_answer_targets_or_iterative_candidate_refits': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, cp, mp, fp, rp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a271v1_plateau_literal_output_geometry.json'
atomic_json(out, report)
if geometries: atomic_torch(out.with_suffix('.pt'), {'geometries': geometries, 'report': report})
print(json.dumps({'eligible_reduced_pool_counts': report['eligible_reduced_pool_counts'],
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
