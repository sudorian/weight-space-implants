"""Joint closed-form native final-Up repair in a fixed token readout metric.

Unchanged Gate factors are bounded using recorded GLU products and linear
Up intervals. No native activation, gradient, iterative fit or model call.
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
base = ROOT / 'experiments/a313v1_duplicate_reference_compression/delta.pt'
comparison = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
ch, comp, ref, fref = [expand(h, p) for p in (base, comparison, reference, format_reference)]
a, host, oldhost, top, stress = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, op, tp, sp)]
assert a['checkpoint_sha256'] == sha(reference) and stress['checkpoint_sha256'] == sha(format_reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256]) and torch.equal(a['inputs'][256:], host['inputs'])
prefix = 'model.layers.17.'
gn, un, dn, pn, pre = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'post_feedforward_layernorm.weight', 'pre_feedforward_layernorm.weight')]
for name in (gn, un, dn, pn, pre):
    assert torch.equal(ch.get(name, h[name]), comp.get(name, h[name])), name
for name in (gn, un, pre):
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
programme = c['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme)]
assert torch.equal(ch[gn][native], h[gn][native]) and torch.equal(ch[un][native], h[un][native])
G0, U0, W0 = [ch.get(n, h[n]).double() for n in (gn, un, dn)]
gain = 1 + ch[pn].double()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

F = a['captures']['17.features'].reshape(-1, 2048)
assert F[:, programme].abs().max() == 0
before32 = a['captures']['17.raw'].reshape(-1, 640)
P = (fnorm(before32) * (1 + ch.get(pre, h[pre]))).double()
before = before32.double()
raw0 = F.double() @ W0.T; rr = rms(raw0)
parent = before + norm(raw0) * gain; pr = rms(parent)
recorded = before + norm(F.double() @ ref[dn].double().T) * (1 + ref[pn].double())
reconstruction = float((recorded - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded.norm())
assert reconstruction < 1e-6
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
tr = rms(teacher); teacher_n = norm(teacher); target = teacher_n - norm(parent)
nt = 384 * 64; nv = 128 * 64
ids = torch.cat([top['topk_ids'].reshape(-1, 8), a['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
tv = top['topk_values'].reshape(-1, 8)
weight = 1 / ((tv[:, 0] - tv[:, 1]).double().clamp_min(0) + .1)
weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
dot_allowance = 650 * eps / (1 - 650 * eps)
counts = [4, 8, 16, 32, 48]; ranks = [4, 8, 16, 32, 64]
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]

def factors(inputs, products, units):
    up = inputs @ U0[units].T
    up_error = dot_allowance * (inputs.abs() @ U0[units].abs().T) + tiny
    gate = inputs @ G0[units].T
    gate_error = dot_allowance * (inputs.abs() @ G0[units].abs().T) + tiny
    lower = torch.minimum(torch.zeros_like(gate), gate - gate_error) * (1 + 4 * eps) - tiny
    upper = torch.maximum(torch.zeros_like(gate), gate + gate_error) * (1 + 4 * eps) + tiny
    safe = up.abs() > up_error
    ferr = eps / (1 - eps) * products.abs() + tiny
    dlo = torch.where(safe, up - up_error, torch.ones_like(up))
    dhi = torch.where(safe, up + up_error, torch.ones_like(up))
    ratios = [(products + s * ferr) / d for s in (-1, 1) for d in (dlo, dhi)]
    rlo = torch.stack(ratios).amin(0); rhi = torch.stack(ratios).amax(0)
    ilo, ihi = torch.maximum(lower, rlo), torch.minimum(upper, rhi)
    inconsistent = safe & (ilo > ihi)
    use_ratio = safe & ~inconsistent
    lower, upper = torch.where(use_ratio, ilo, lower), torch.where(use_ratio, ihi, upper)
    return (lower + upper) / 2, (upper - lower) / 2, up, up_error, up.abs() > 2 * up_error, int(inconsistent.sum())

# One canonical input basis and one original-target unit ranking, before fits.
X = P[:nt] / rr[:nt] / pr[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
C = Z.T @ Z / nt
ce, cv = torch.linalg.eigh((C + C.T) / 2); ce = ce.clamp_min(0)
inverse_root = (cv / (ce + .001).sqrt()[None]) @ cv.T
cross = inverse_root @ (Z.T @ target[:nt] / nt)
canonical, spectrum, _ = torch.linalg.svd(cross, full_matrices=False)
input_map = inverse_root @ canonical[:, :max(ranks)] / sd[:, None]
features = X @ input_map
priority = torch.full((len(native),), -torch.inf, dtype=torch.float64)
stable_fraction = torch.zeros(len(native), dtype=torch.float64)
inconsistencies = 0
for start in range(0, len(native), 64):
    units = native[start:start + 64]
    phi, error, up, up_error, stable, bad = factors(P[:nt], F[:nt, units].double(), units)
    inconsistencies += bad
    frac = stable.double().mean(0); stable_fraction[start:start + len(units)] = frac
    WM = W0[:, units] * gain[:, None]
    unit_norm = WM.norm(dim=0).clamp_min(1e-15)
    projected = target[:nt] @ (WM / unit_norm[None])
    cross_rank = Z.T @ (phi * stable * projected) / nt
    solved = cv @ ((cv.T @ cross_rank) / (ce + .001)[:, None])
    score = (cross_rank * solved).sum(0).clamp_min(0) / (phi.square() * stable).mean(0).clamp_min(1e-15)
    priority[start:start + len(units)] = torch.where(frac >= .99, score, -torch.inf)
assert int((stable_fraction >= .99).sum()) >= max(counts)
order = priority.argsort(descending=True); units = native[order[:max(counts)]]
phi, phi_error, old_up, old_up_error, stable, chosen_bad = factors(P, F[:, units].double(), units)
unit_norm = (W0[:, units] * gain[:, None]).norm(dim=0).clamp_min(1e-15)
directions = W0[:, units] * gain[:, None] / unit_norm[None]
train_phi = phi[:nt] * stable[:nt]
del X, Z, C, ce, cv, inverse_root, cross, canonical, projected, cross_rank, solved, WM
del error, up, up_error

# Fixed per-token metric, including every cross-unit block.
H = torch.empty(nt, max(counts), max(counts), dtype=torch.float64)
b = torch.empty(nt, max(counts), dtype=torch.float64)
probe = torch.sin(torch.arange(max(counts) * max(ranks), dtype=torch.float64)).reshape(max(counts), max(ranks)) * .001
probe_score_energy = 0.; probe_target_inner = 0.
for start in range(0, nt, 256):
    stop = min(start + 256, nt)
    vectors = E[ids[start:stop]].double() * fg; vectors -= vectors.mean(1, keepdim=True)
    reduced = vectors @ directions
    response = (vectors * target[start:stop, None]).sum(-1)
    H[start:stop] = reduced.transpose(1, 2) @ reduced / ids.shape[1] * weight[start:stop, None, None]
    b[start:stop] = (reduced * response[:, :, None]).mean(1) * weight[start:stop, None]
    coefficient = (features[start:stop] @ probe.T) * train_phi[start:stop]
    probe_score = (reduced * coefficient[:, None]).sum(-1)
    probe_score_energy += float((probe_score.square().mean(1) * weight[start:stop]).sum()) / nt
    probe_target_inner += float(((probe_score * response).mean(1) * weight[start:stop]).sum()) / nt
gram_full = torch.empty(max(counts), max(ranks), max(counts), max(ranks), dtype=torch.float64)
for i in range(max(counts)):
    for j in range(i, max(counts)):
        mass = H[:, i, j] * train_phi[:, i] * train_phi[:, j]
        block = features.T @ (features * mass[:, None]) / nt
        gram_full[i, :, j, :] = block; gram_full[j, :, i, :] = block.T
rhs_full = (b * train_phi).T @ features / nt
flat_probe = probe.reshape(-1)
probe_gram = float(flat_probe @ gram_full.reshape(len(flat_probe), len(flat_probe)) @ flat_probe)
probe_rhs = float((probe * rhs_full).sum())
assert abs(probe_gram - probe_score_energy) <= 1e-8 * max(1., abs(probe_score_energy))
assert abs(probe_rhs - probe_target_inner) <= 1e-8 * max(1., abs(probe_target_inner))
del H, b, features, vectors, reduced, response, train_phi, coefficient, probe_score
bm = json.loads(base.with_suffix('.manifest.json').read_text())

def candidate_state(inputs, original_raw, original_before, factors_, factor_error_, old_up_, old_error_, chosen_units, proposed):
    delta = proposed.double() - U0[chosen_units]
    delta_up = inputs @ delta.T
    raw = original_raw + (factors_ * delta_up) @ W0[:, chosen_units].T
    state = original_before + norm(raw) * gain
    new_up = inputs @ proposed.double().T
    new_error = dot_allowance * (inputs.abs() @ proposed.double().abs().T) + tiny
    phi_upper = factors_.abs() + factor_error_
    feature_error = factor_error_ * delta_up.abs() + phi_upper * (new_error + old_error_)
    feature_error += eps / (1 - eps) * phi_upper * (new_up.abs() + new_error + old_up_.abs() + old_error_) + 2 * tiny
    raw_error = feature_error @ W0[:, chosen_units].abs().T
    radius = raw_error.norm(dim=1); raw_norm = raw.norm(dim=1)
    denominator = (raw_norm - radius).clamp_min(1e-30)
    post_bound = 640 ** .5 * gain.abs() * (raw_error / denominator[:, None]
        + raw.abs() * (radius / (raw_norm.clamp_min(1e-30) * denominator))[:, None])
    return state, raw, post_bound

def quality(state, raw, bound):
    matches = certified = 0; score_error = 0.
    direction = norm(state) - teacher_n[nt:]
    for start in range(0, nv, 256):
        stop = min(start + 256, nv); local_ids = ids[nt+start:nt+stop]
        vectors = E[local_ids].double() * fg
        scores = (vectors * state[start:stop, None]).sum(-1)
        winner = local_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
        good = winner == truth[nt+start:nt+stop]; matches += int(good.sum())
        desired = E[truth[nt+start:nt+stop]].double() * fg
        difference = desired[:, None] - vectors
        margin = (difference * state[start:stop, None]).sum(-1)
        margin -= (difference.abs() * bound[start:stop, None]).sum(-1)
        margin[local_ids == truth[nt+start:nt+stop, None]] = torch.inf
        certified += int((good & (margin.amin(1) > 0)).sum())
        vectors -= vectors.mean(1, keepdim=True)
        err = (vectors * direction[start:stop, None]).sum(-1)
        score_error += float((err.square().mean(1) * weight[nt+start:nt+stop]).sum())
    return {'validation_restricted_matches': matches, 'validation_certified_restricted_matches': certified,
        'validation_readout_MSE': score_error / float(weight[nt:].sum()),
        'validation_paired_HOST_state_relative_RMS': float(((state - teacher[nt:]) / tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_RMS': float(direction.square().mean().sqrt()),
        'maximum_raw_RMS_relative_change': float((rms(raw) / rr[nt:] - 1).abs().max()),
        'factor_and_Up_roundoff_relative_state_RMS_bound': float((bound / tr[nt:]).square().mean().sqrt())}

baseline = quality(parent[nt:], raw0[nt:], torch.zeros_like(parent[nt:]))
assert baseline['validation_restricted_matches'] == 8003
print(json.dumps({'baseline': baseline, 'stable_native_units': int((stable_fraction >= .99).sum()),
    'ranked_units': units.tolist(), 'canonical_spectrum': spectrum[:16].tolist(),
    'quadratic_probe': [probe_gram, probe_score_energy, probe_rhs, probe_target_inner]}), flush=True)
trials = []; best = None; diagnostic = None
for count in counts:
    chosen = units[:count]
    for rank in ranks:
        gram = gram_full[:count, :rank, :count, :rank].reshape(count * rank, count * rank)
        scale = gram.diagonal().clamp_min(1e-30).sqrt()
        standardized = gram / scale[:, None] / scale[None]
        ev, vec = torch.linalg.eigh((standardized + standardized.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (rhs_full[:count, :rank].reshape(-1) / scale)
        for ridge in ridges:
            beta = ((vec @ (rhs / (ev + ridge))) / scale).reshape(count, rank)
            correction = (beta @ input_map[:, :rank].T) / unit_norm[:count, None]
            for strength in strengths:
                proposed = (U0[chosen] + strength * correction).float()
                state, raw, bound = candidate_state(P[nt:], raw0[nt:], before[nt:], phi[nt:, :count],
                    phi_error[nt:, :count], old_up[nt:, :count], old_up_error[nt:, :count], chosen, proposed)
                q = quality(state, raw, bound)
                changed_before = int((ch[un][chosen].view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                changed_after = int((proposed.view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                total = bm['changed_scalars'] - changed_before + changed_after
                rec = {'native_unit_count': count, 'canonical_input_rank': rank, 'ridge': ridge, 'strength': strength,
                    'units': chosen.tolist(), **q, 'total_changed_scalars': total, 'maximum_Up_weight': float(proposed.abs().max()),
                    'unstable_training_observations': int((~stable[:nt, :count]).sum()),
                    'unstable_validation_observations': int((~stable[nt:, :count]).sum())}
                rec['eligible'] = bool(proposed.isfinite().all() and proposed.abs().max() <= 1e6 and total <= 5016268
                    and q['validation_restricted_matches'] >= baseline['validation_restricted_matches'] + 16
                    and q['validation_certified_restricted_matches'] >= baseline['validation_restricted_matches'] + 8
                    and q['validation_paired_HOST_state_relative_RMS'] <= 1.05 * baseline['validation_paired_HOST_state_relative_RMS']
                    and q['validation_paired_HOST_direction_RMS'] <= 1.05 * baseline['validation_paired_HOST_direction_RMS']
                    and q['maximum_raw_RMS_relative_change'] < .05)
                trials.append(rec)
                key = (q['validation_restricted_matches'], -q['validation_readout_MSE'])
                if diagnostic is None or key > diagnostic[0]: diagnostic = key, rec
                if rec['eligible'] and (best is None or key > best[0]):
                    best = key, chosen.clone(), proposed.clone(), state.clone(), rec
        print(json.dumps({'unit_count': count, 'input_rank': rank, 'best_diagnostic': diagnostic[-1],
            'selected': None if best is None else best[-1], 'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)

def full_readout(state, expected, bound=None):
    x = (norm(state) * fg).float(); maximum = torch.full((len(x),), -torch.inf)
    winner = torch.full_like(expected, -1); competitor = torch.full_like(maximum, -torch.inf)
    desired = (x * E[expected]).sum(1)
    certified_margin = torch.full((len(x),), torch.inf, dtype=torch.float64)
    for start in range(0, len(E), 2048):
        logits = x @ E[start:start+2048].T
        values, indices = logits.max(1); better = values > maximum
        maximum[better] = values[better]; winner[better] = start + indices[better]
        local = expected - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf; competitor = torch.maximum(competitor, logits.amax(1))
        if bound is not None:
            # Triangle bound on |E_expected-E_competitor| keeps allocation small.
            vectors = E[start:start+2048].double() * fg
            desired_vectors = E[expected].double() * fg
            margin = (state * desired_vectors).sum(1, keepdim=True) - state @ vectors.T
            margin -= (bound * desired_vectors.abs()).sum(1, keepdim=True) + bound @ vectors.abs().T
            margin[rows, local[rows]] = torch.inf
            certified_margin = torch.minimum(certified_margin, margin.amin(1))
    return {'tokens': len(x), 'matches': int((winner == expected).sum()),
        'minimum_target_margin': float((desired - competitor).min()),
        'minimum_certified_raw_margin': None if bound is None else float(certified_margin.min())}

full = fmt = provisional = None
if best is not None:
    provisional = best[-1]
    full = {'parent': full_readout(parent[nt:], truth[nt:]), 'candidate': full_readout(best[3], truth[nt:])}
    assert full['parent']['matches'] == 8003
    if full['candidate']['matches'] < full['parent']['matches'] + 16: best = None
if best is not None:
    chosen, proposed = best[1:3]
    SF = stress['captures']['17.features'][:, 8:13].reshape(-1, 2048).double()
    SB32 = stress['captures']['17.raw'][:, 8:13].reshape(-1, 640)
    SP = (fnorm(SB32) * (1 + ch.get(pre, h[pre]))).double(); SB = SB32.double()
    Sraw = SF @ W0.T
    sf, se, su, sue, ss, sbad = factors(SP, SF[:, chosen], chosen)
    state, raw, bound = candidate_state(SP, Sraw, SB, sf, se, su, sue, chosen, proposed)
    expected = stress['topk_ids'][:, :, 0].reshape(-1).long()
    fmt = {'parent': full_readout(SB + norm(Sraw) * gain, expected),
        'candidate': full_readout(state, expected, bound),
        'maximum_raw_RMS_relative_change': float((rms(raw) / rms(Sraw) - 1).abs().max()),
        'interval_inconsistencies': sbad}
    assert fmt['parent']['matches'] == len(expected)
    if (fmt['candidate']['matches'] != len(expected) or fmt['candidate']['minimum_target_margin'] <= 1
            or fmt['candidate']['minimum_certified_raw_margin'] <= 0 or fmt['maximum_raw_RMS_relative_change'] >= .02): best = None

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'comparison_checkpoint': str(comparison.relative_to(ROOT)), 'training_windows': 384, 'holdback_windows': 128,
    'count_trials': counts, 'input_rank_trials': ranks, 'ridge_trials': ridges, 'strength_trials': strengths,
    'canonical_whitening_ridge': .001, 'stable_training_fraction_required': .99,
    'stable_native_units': int((stable_fraction >= .99).sum()), 'ranked_units': units.tolist(),
    'training_interval_inconsistencies': inconsistencies, 'chosen_interval_inconsistencies': chosen_bad,
    'original_capture_state_reconstruction_relative_error': reconstruction,
    'quadratic_probe': {'gram': probe_gram, 'explicit_energy': probe_score_energy, 'rhs': probe_rhs, 'explicit_target_inner': probe_target_inner},
    'baseline': baseline, 'trials': trials, 'best_diagnostic': diagnostic[-1], 'provisional': provisional,
    'full_vocabulary_validation': full, 'format_validation': fmt, 'selected': None if best is None else best[-1],
    'target': 'Original paired HOST normalized final state minus original A142 parent, fixed original branch and final RMS; all solves use this original target',
    'metric': 'Centered original HOST/A123 top8 union, finalnormgain, 1/(HOSTgap+0.1) training-mean-normalized and capped100; joint cross-unit normal equations',
    'factor_scope': 'Recorded native factors inferred from products/linear Up intervals, no native activation or derivative; bounds cover factor and Up rounding only',
    'trajectory_scope': 'Earlier A123 recorded final inputs; A313 upstream edits change the unobserved final trajectory; official score required',
    'all_other_weights_unchanged': True, 'no_gradients_native_activation_model_execution_answers_or_iterative_refits': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', comparison, reference, format_reference, ap, hp, op, tp, sp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a347v1_joint_token_metric_final_up.json'; atomic_json(out, report)
if best is not None:
    Unew = ch[un].clone(); Unew[best[1]] = best[2]
    atomic_torch(out.with_suffix('.pt'), {'up': Unew, 'report': report})
print(json.dumps({'selected': report['selected'], 'full': full, 'format': fmt,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
