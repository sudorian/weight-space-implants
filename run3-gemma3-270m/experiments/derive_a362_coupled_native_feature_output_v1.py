"""Coupled native input-feature and original-target output construction.

Unchanged Gate factors use recorded GLU products and matched-shape float32
Up arithmetic, conditional on A356 controls. No native activation, gradient,
iterative fit or model call.
"""
import json
import resource
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
comparison = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
audit_path=ROOT/'artifacts/a356v1_recorded_linear_kernel_identity.json'
audit=json.loads(audit_path.read_text())
assert audit['all_available_controls_bit_exact_and_intervals_consistent']
for p,digest in {**audit['sources'],**audit['inputs']}.items():assert sha(ROOT/p)==digest,p
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
word_batch=a['request']['batch_size'];word_length=a['inputs'].shape[1]
word_raw=before32.reshape(len(a['inputs']),word_length,640)
word_pre=torch.cat([fnorm(word_raw[i:i+word_batch])*(1+ch.get(pre,h[pre])) for i in range(0,len(word_raw),word_batch)])
P=word_pre.reshape(-1,640).double()
before = before32.double()
raw0 = F.double() @ W0.T; rr = rms(raw0)
parent = before + norm(raw0) * gain; pr = rms(parent)
recorded = before + norm(F.double() @ ref[dn].double().T) * (1 + ref[pn].double())
reconstruction = float((recorded - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded.norm())
assert reconstruction < 1e-6
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
tr = rms(teacher); teacher_n = norm(teacher); target = teacher_n - norm(parent)
del recorded, ref, fref, comp
nt = 384 * 64; nv = 128 * 64
ids = torch.cat([top['topk_ids'].reshape(-1, 8), a['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
tv = top['topk_values'].reshape(-1, 8)
weight = 1 / ((tv[:, 0] - tv[:, 1]).double().clamp_min(0) + .1)
weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
dot_allowance = 650 * eps / (1 - 650 * eps)
counts = [16, 32, 64, 128, 256]
up_fractions = [0., .1, .25, .5, 1.]
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]

def linear_all(inputs, weight, batch):
    assert inputs.dtype==weight.dtype==torch.float32 and inputs.ndim==3
    return torch.cat([functional.linear(inputs[i:i+batch],weight) for i in range(0,len(inputs),batch)])

# Conditional matched-shape linear identity; this is not native GELU replay.
up_cache=linear_all(word_pre,ch[un],word_batch).reshape(-1,2048)
gate_cache=linear_all(word_pre,ch[gn],word_batch).reshape(-1,2048)
assert torch.equal(gate_cache[:,programme].view(torch.int32),a['captures']['17.program_gates'].reshape(-1,len(programme)).view(torch.int32))

def factor_intervals(up32,gate32,products):
    up,gate,products=up32.double(),gate32.double(),products.double()
    safe=up!=0
    assert not ((~safe)&(products!=0)).any()
    ferr=eps/(1-eps)*products.abs()+tiny
    divisor=torch.where(safe,up,torch.ones_like(up))
    lower=torch.minimum((products-ferr)/divisor,(products+ferr)/divisor)
    upper=torch.maximum((products-ferr)/divisor,(products+ferr)/divisor)
    lo=torch.minimum(gate,torch.zeros_like(gate))*(1+4*eps)-tiny
    hi=torch.maximum(gate,torch.zeros_like(gate))*(1+4*eps)+tiny
    lower=torch.where(safe,torch.maximum(lower,lo),lo)
    upper=torch.where(safe,torch.minimum(upper,hi),hi)
    bad=int((lower>upper).sum());assert bad==0
    return (lower+upper)/2,(upper-lower)/2,safe,bad

# Every altered Up belongs to the existing complete donor rectangle.
donors = torch.as_tensor(c['native_donor_repairs']['17']['donors']).long()
pool = torch.as_tensor(c['output_pool']).long()
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), pool)]
assert len(donors) == 512 and len(pool) == 96 and len(ordinary) == 544
assert not torch.isin(donors, programme).any() and len(donors.unique()) == len(donors)
assert (gain[pool] == 1e6).all() and (gain != 0).all()
donor_native_positions = torch.tensor([int((native == u).nonzero().item()) for u in donors])
phi, phi_error, stable, factor_bad = factor_intervals(up_cache[:, donors], gate_cache[:, donors], F[:, donors])
del up_cache, gate_cache
assert stable[:nt].double().mean(0).min() >= .99

# Fixed original readout metric and response subspace, independent of all fits.
metric = torch.zeros(640, 640, dtype=torch.float64)
for start in range(0, nt, 256):
    stop = min(start + 256, nt)
    vectors = E[ids[start:stop]].double() * fg
    vectors -= vectors.mean(1, keepdim=True)
    vectors *= weight[start:stop, None, None].sqrt()
    flat = vectors.reshape(-1, 640)
    metric.add_(flat.T @ flat)
metric /= float(weight[:nt].sum()) * ids.shape[1]
me, mv = torch.linalg.eigh((metric + metric.T) / 2)
metric_root = (mv * me.clamp_min(0).sqrt()[None]) @ mv.T
weighted_response = target[:nt] @ metric_root
response_gram = weighted_response.T @ weighted_response / nt
response_values, response_vectors = torch.linalg.eigh((response_gram + response_gram.T) / 2)
response_map = metric_root @ response_vectors[:, -16:]
response = target[:nt] @ response_map
response_spectrum = response_values[-16:].flip(0).tolist()
del vectors, flat, me, mv, metric_root, weighted_response, response_gram, response_values, response_vectors

X = P[:nt] / rr[:nt] / pr[:nt]
weight_sum = float(weight[:nt].sum())
fields = torch.empty((len(donors), 640), dtype=torch.float64)
priorities = torch.empty(len(donors), dtype=torch.float64)
field_diagnostics = []
for i, unit in enumerate(donors.tolist()):
    design = X * (phi[:nt, i] * stable[:nt, i])[:, None]
    sd = ((design.square() * weight[:nt, None]).sum(0) / weight_sum).sqrt().clamp_min(1e-15)
    Z = design / sd
    gram = Z.T @ (Z * weight[:nt, None]) / weight_sum
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    inverse_root = (vectors / (values.clamp_min(0) + .001).sqrt()[None]) @ vectors.T
    cross = inverse_root @ (Z.T @ (response * weight[:nt, None]) / weight_sum)
    left, singular, _ = torch.linalg.svd(cross, full_matrices=False)
    field = (inverse_root @ left[:, 0]) / sd
    product = (P[:nt] @ field) * phi[:nt, i]
    original = F[:nt, unit].double()
    sign = 1. if float((product * original * weight[:nt]).sum()) >= 0 else -1.
    original_rms = ((original.square() * weight[:nt]).sum() / weight_sum).sqrt()
    new_rms = ((product.square() * weight[:nt]).sum() / weight_sum).sqrt().clamp_min(1e-30)
    field *= sign * original_rms / new_rms
    fields[i] = field; priorities[i] = singular[0].square()
    field_diagnostics.append({'unit': unit, 'original_response_priority': float(priorities[i]),
        'physical_feature_RMS': float(original_rms), 'maximum_field_weight': float(field.abs().max())})
    if (i + 1) % 64 == 0:
        print(json.dumps({'stage': 'fixed_original_response_fields', 'completed': i + 1,
            'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
del X, design, Z, sd, gram, values, vectors, inverse_root, cross, left, singular, field, product, original, P
order = priorities.argsort(descending=True)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
val_ids = ids[nt:]; val_truth = truth[nt:]
val_difference = E[val_ids].double() * fg
wanted_vectors = val_difference[:, 0].clone()
val_difference.neg_().add_(wanted_vectors[:, None])
val_abs_difference = val_difference.abs()
val_same = val_ids == val_truth[:, None]
del wanted_vectors

def selected_up(inputs, full_up, units, positions=64, batch_size=None):
    batch_size = word_batch if batch_size is None else batch_size
    return torch.cat([functional.linear(inputs[i:i+batch_size], full_up)[:, :positions, units]
        for i in range(0, len(inputs), batch_size)]).reshape(-1, len(units)).double()

def post_bound(raw, error):
    radius = error.norm(dim=1); raw_norm = raw.norm(dim=1)
    denominator = (raw_norm - radius).clamp_min(1e-30)
    return 640 ** .5 * gain.abs() * (error / denominator[:, None]
        + raw.abs() * (radius / (raw_norm.clamp_min(1e-30) * denominator))[:, None])

def quality(state, raw, bound):
    margin = torch.einsum('nkd,nd->nk', val_difference, state)
    winner = val_ids.gather(1, margin.argmin(1, keepdim=True)).flatten()
    good = winner == val_truth
    lower = margin - torch.einsum('nkd,nd->nk', val_abs_difference, bound)
    lower[val_same] = torch.inf
    direction = norm(state) - teacher_n[nt:]
    score = torch.einsum('nkd,nd->nk', val_difference, direction)
    score -= score.mean(1, keepdim=True)
    return {'validation_restricted_matches': int(good.sum()),
        'validation_certified_restricted_matches': int((good & (lower.amin(1) > 0)).sum()),
        'validation_readout_MSE': float((score.square().mean(1) * weight[nt:]).sum() / weight[nt:].sum()),
        'validation_paired_HOST_state_relative_RMS': float(((state - teacher[nt:]) / tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_RMS': float(direction.square().mean().sqrt()),
        'maximum_raw_RMS_relative_change': float((rms(raw) / rr[nt:] - 1).abs().max()),
        'conditional_factor_and_product_relative_state_RMS_bound': float((bound / tr[nt:]).square().mean().sqrt())}

baseline = quality(parent[nt:], raw0[nt:], torch.zeros_like(parent[nt:]))
assert baseline['validation_restricted_matches'] == 8003
control_features = F[nt:].double()
control_up = selected_up(word_pre[384:], ch[un], donors)
control_features[:, donors] = phi[nt:] * control_up
control_raw = control_features @ W0.T
control_error = phi_error[nt:] * control_up.abs()
control_error += eps / (1 - eps) * (phi[nt:].abs() + phi_error[nt:]) * control_up.abs() + tiny
control_bound = post_bound(control_raw, control_error @ W0[:, donors].abs().T)
control = quality(before[nt:] + norm(control_raw) * gain, control_raw, control_bound)
control['maximum_raw_relative_RMS_reconstruction_error'] = float(((control_raw - raw0[nt:]) / rr[nt:]).square().mean(1).sqrt().max())
assert control['validation_restricted_matches'] == 8003 and control['maximum_raw_relative_RMS_reconstruction_error'] < 1e-6
del control_features, control_up, control_raw, control_error, control_bound
print(json.dumps({'baseline': baseline, 'control': control}), flush=True)

# The target and prior are fixed before any altered feature map is fitted.
target_fixed = teacher_n[:nt] - before[:nt] / pr[:nt]
prior_B = W0 * gain[:, None]
trials = []; best = None; diagnostic = None; normal_equation_probes = []
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 + 3_000_000_000 < 11_500_000_000
for count in counts:
    indices = order[:count]; chosen = donors[indices]
    for fraction in up_fractions:
        new_up = ch[un].clone()
        new_up[chosen] = (U0[chosen] + fraction * fields[indices]).float()
        actual_up = selected_up(word_pre, new_up, chosen)
        products = phi[:, indices] * actual_up
        train_design = F[:nt].double(); val_features = F[nt:].double()
        train_design[:, chosen] = products[:nt]
        val_features[:, chosen] = products[nt:]
        train_design.div_(rr[:nt]).div_(pr[:nt])
        residual_from_fixed_prior = target_fixed - train_design @ prior_B.T
        standard = ((train_design[:, native].square() * weight[:nt, None]).sum(0) / weight_sum).sqrt().clamp_min(1e-15)
        Z = train_design[:, native] / standard
        covariance = Z.T @ (Z * weight[:nt, None]) / weight_sum
        rhs_strong = Z.T @ (residual_from_fixed_prior[:, pool] * weight[:nt, None]) / weight_sum
        rhs_ordinary = Z[:, donor_native_positions].T @ (residual_from_fixed_prior[:, ordinary] * weight[:nt, None]) / weight_sum
        donor_covariance = covariance[donor_native_positions][:, donor_native_positions]
        probe = torch.sin(torch.arange(len(native), dtype=torch.float64)) * .001
        output_probe = torch.cos(torch.arange(len(pool), dtype=torch.float64))
        response_probe = Z @ probe
        gram_probe = float(probe @ covariance @ probe)
        explicit_gram = float((response_probe.square() * weight[:nt]).sum() / weight_sum)
        rhs_probe = float(probe @ rhs_strong @ output_probe)
        explicit_rhs = float((response_probe * (residual_from_fixed_prior[:, pool] @ output_probe) * weight[:nt]).sum() / weight_sum)
        assert abs(gram_probe - explicit_gram) < 1e-8 * max(1., abs(explicit_gram))
        assert abs(rhs_probe - explicit_rhs) < 1e-8 * max(1., abs(explicit_rhs))
        normal_equation_probes.append({'count': count, 'Up_fraction': fraction,
            'gram': gram_probe, 'explicit_energy': explicit_gram, 'rhs': rhs_probe, 'explicit_target_inner': explicit_rhs})
        del probe, output_probe, response_probe
        en, vn = torch.linalg.eigh((covariance + covariance.T) / 2); en = en.clamp_min(0)
        ed, vd = torch.linalg.eigh((donor_covariance + donor_covariance.T) / 2); ed = ed.clamp_min(0)
        projected_strong = vn.T @ rhs_strong
        projected_ordinary = vd.T @ rhs_ordinary
        del train_design, residual_from_fixed_prior, Z, covariance, rhs_strong, rhs_ordinary, donor_covariance
        feature_error = phi_error[nt:, indices] * actual_up[nt:].abs()
        feature_error += eps / (1 - eps) * (phi[nt:, indices].abs() + phi_error[nt:, indices]) * actual_up[nt:].abs() + tiny
        del actual_up, products
        up_changed = int((new_up.view(torch.int32) != h[un].view(torch.int32)).sum())
        for ridge in ridges:
            correction_B = torch.zeros_like(W0)
            strong = (vn @ (projected_strong / (en + ridge)[:, None])) / standard[:, None]
            donor_fit = (vd @ (projected_ordinary / (ed + ridge)[:, None])) / standard[donor_native_positions, None]
            correction_B[pool[:, None], native[None]] = strong.T
            correction_B[ordinary[:, None], donors[None]] = donor_fit.T
            correction = correction_B / gain[:, None]
            assert correction[:, programme].abs().max() == 0
            for strength in strengths:
                new_down = (W0 + strength * correction).float()
                assert torch.equal(new_down[:, programme], ch[dn][:, programme])
                raw = val_features @ new_down.double().T
                state = before[nt:] + norm(raw) * gain
                bound = post_bound(raw, feature_error @ new_down[:, chosen].double().abs().T)
                q = quality(state, raw, bound)
                down_changed = int((new_down.view(torch.int32) != h[dn].view(torch.int32)).sum())
                total = bm['changed_scalars'] - bm['per_tensor'][un] - bm['per_tensor'][dn] + up_changed + down_changed
                rec = {'native_unit_count': count, 'units': chosen.tolist(), 'Up_fraction': fraction,
                    'Down_ridge': ridge, 'Down_strength': strength, **q,
                    'changed_Up_scalars': up_changed, 'changed_Down_scalars': down_changed,
                    'total_changed_scalars': total, 'maximum_native_Up_weight': float(new_up[chosen].abs().max()),
                    'maximum_Down_weight': float(new_down.abs().max())}
                rec['eligible'] = bool(new_up.isfinite().all() and new_down.isfinite().all()
                    and new_up.abs().max() <= 1e6 and new_down.abs().max() <= 1e6 and total <= 5016268
                    and q['validation_restricted_matches'] >= baseline['validation_restricted_matches'] + 16
                    and q['validation_certified_restricted_matches'] >= baseline['validation_restricted_matches'] + 8
                    and q['validation_paired_HOST_state_relative_RMS'] <= 1.05 * baseline['validation_paired_HOST_state_relative_RMS']
                    and q['validation_paired_HOST_direction_RMS'] <= 1.05 * baseline['validation_paired_HOST_direction_RMS']
                    and q['maximum_raw_RMS_relative_change'] < .05)
                trials.append(rec)
                key = (q['validation_restricted_matches'], -q['validation_readout_MSE'])
                if diagnostic is None or key > diagnostic[0]: diagnostic = key, rec
                if rec['eligible'] and (best is None or key > best[0]):
                    best = key, chosen.clone(), new_up.clone(), new_down.clone(), state.clone(), rec
        print(json.dumps({'count': count, 'Up_fraction': fraction,
            'best_diagnostic': {k:v for k,v in diagnostic[-1].items() if k != 'units'},
            'selected': None if best is None else {k:v for k,v in best[-1].items() if k != 'units'},
            'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 12_000_000_000
        del new_up, val_features, standard, en, vn, ed, vd, projected_strong, projected_ordinary
        del feature_error, correction_B, strong, donor_fit, correction, new_down, raw, state, bound
del val_difference, val_abs_difference, target_fixed, prior_B



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
    full = {'parent': full_readout(parent[nt:], truth[nt:]), 'candidate': full_readout(best[4], truth[nt:])}
    assert full['parent']['matches'] == 8003
    if full['candidate']['matches'] < full['parent']['matches'] + 16: best = None
if best is not None:
    chosen, proposed_up, proposed_down = best[1:4]
    SF = stress['captures']['17.features'][:, 8:13].reshape(-1, 2048).double()
    SB = stress['captures']['17.raw'][:, 8:13].reshape(-1, 640).double()
    SP = stress['captures']['17.pre']; format_batch = stress['request']['batch_size']
    SU = linear_all(SP, ch[un], format_batch)[:, 8:13].reshape(-1, 2048)
    SG = linear_all(SP, ch[gn], format_batch)[:, 8:13].reshape(-1, 2048)
    sf, se, ss, sbad = factor_intervals(SU[:, chosen], SG[:, chosen], SF[:, chosen])
    new_values = selected_up(SP, proposed_up, chosen, 14, format_batch).reshape(512, 14, len(chosen))[:, 8:13].reshape(-1, len(chosen))
    Sraw = SF @ W0.T
    updated_features = SF.clone(); updated_features[:, chosen] = sf * new_values
    raw = updated_features @ proposed_down.double().T
    state = SB + norm(raw) * gain
    feature_error = se * new_values.abs()
    feature_error += eps / (1 - eps) * (sf.abs() + se) * new_values.abs() + tiny
    bound = post_bound(raw, feature_error @ proposed_down[:, chosen].double().abs().T)
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
    'Up_count_trials': counts, 'Up_fraction_trials': up_fractions,
    'Down_ridge_trials': ridges, 'Down_strength_trials': strengths,
    'fixed_input_covariance_whitening_ridge': .001, 'fixed_original_response_rank': 16,
    'donor_pool': donors.tolist(), 'ranked_donor_units': donors[order].tolist(),
    'field_diagnostics': field_diagnostics, 'original_response_spectrum': response_spectrum,
    'training_interval_inconsistencies': factor_bad,
    'original_capture_state_reconstruction_relative_error': reconstruction,
    'baseline': baseline, 'factor_control': control, 'normal_equation_probes': normal_equation_probes, 'trials': trials, 'best_diagnostic': diagnostic[-1],
    'provisional': provisional, 'full_vocabulary_validation': full, 'format_validation': fmt,
    'selected': None if best is None else best[-1],
    'target': 'Original paired HOST normalized final state minus original pre17 divided by original parent final RMS. All normalizers, feature directions and ridge priors frozen before output solves.',
    'construction': 'One original-response canonical direction per unchanged native Gate, followed by one native output ridge-with-prior solve for each fixed declared feature map. No output fit feeds a new feature direction or residual fit.',
    'output_support': 'Existing512native donors for544ordinary rows; all2037native features for96strong rows. Programme columns, all Gate weights and gains fixed.',
    'factor_scope': 'Recorded products divided by matched-shape float32 Up, conditional on A356 deterministic controls. Bounds cover factor and product rounding; complete Down/RMS/readout floating rounding and changed upstream trajectories are outside this component certificate.',
    'trajectory_scope': 'Original A123 recorded final inputs; earlier A354 edits change its unobserved final trajectory. Official scoring remains required.',
    'no_gradients_native_activation_model_execution_answers_or_iterative_refits': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a355_expanded_joint_native_up_v1.py': sha(ROOT / 'experiments/derive_a355_expanded_joint_native_up_v1.py'),
        'experiments/analyze_a356_recorded_linear_kernel_identity_v1.py': sha(ROOT / 'experiments/analyze_a356_recorded_linear_kernel_identity_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', comparison, reference, format_reference, ap, hp, op, tp, sp, audit_path)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a362v1_coupled_native_feature_output.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'up': best[2], 'down': best[3], 'report': report})
print(json.dumps({'selected': report['selected'], 'full': full, 'format': fmt,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)

