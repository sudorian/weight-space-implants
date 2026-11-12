"""Expanded joint native Up repair with a fixed global readout metric.

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
counts = [64, 128, 256, 384, 512]; ranks = [16]
oracle_ridges = [1e-4, .001, .01, .1, 1.]
output_rank = 16
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

# One canonical input basis and one original-target unit ranking, before fits.
X = P[:nt] / rr[:nt] / pr[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
C = Z.T @ Z / nt
ce, cv = torch.linalg.eigh((C + C.T) / 2); ce = ce.clamp_min(0)
inverse_root = (cv / (ce + .001).sqrt()[None]) @ cv.T
cross = inverse_root @ (Z.T @ target[:nt] / nt)
canonical, spectrum, _ = torch.linalg.svd(cross, full_matrices=False)
input_map = inverse_root @ canonical[:, :max(ranks)] / sd[:, None]
features = X  # All 640 physical directions, using frozen original RMS scales.
priority = torch.full((len(native),), -torch.inf, dtype=torch.float64)
stable_fraction = torch.zeros(len(native), dtype=torch.float64)
inconsistencies = 0
for start in range(0, len(native), 64):
    units = native[start:start + 64]
    phi, error, stable, bad = factor_intervals(up_cache[:nt,units],gate_cache[:nt,units],F[:nt,units])
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
phi, phi_error, stable, chosen_bad = factor_intervals(up_cache[:,units],gate_cache[:,units],F[:,units])
observed_selected=F[:,units].double()
del up_cache,gate_cache
unit_norm = (W0[:, units] * gain[:, None]).norm(dim=0).clamp_min(1e-15)
directions = W0[:, units] * gain[:, None] / unit_norm[None]
train_phi = phi[:nt] * stable[:nt]
del X, Z, C, ce, cv, inverse_root, cross, canonical, projected, cross_rank, solved, WM, P
del error

# Fixed positive-semidefinite global metric from original training readouts.
metric=torch.zeros(640,640,dtype=torch.float64)
for start in range(0,nt,256):
    stop=min(start+256,nt)
    vectors=E[ids[start:stop]].double()*fg
    vectors-=vectors.mean(1,keepdim=True)
    vectors*=weight[start:stop,None,None].sqrt()
    flat=vectors.reshape(-1,640)
    metric.add_(flat.T@flat)
metric/=float(weight[:nt].sum())*ids.shape[1]
metric=(metric+metric.T)/2
# Fixed metric-weighted response PCA; no candidate output enters this basis.
me, mv = torch.linalg.eigh(metric)
metric_root = (mv * me.clamp_min(0).sqrt()[None]) @ mv.T
weighted_target = target[:nt] @ metric_root
response_gram = weighted_target.T @ weighted_target / nt
re, rv = torch.linalg.eigh((response_gram + response_gram.T) / 2)
output_map = metric_root @ rv[:, -output_rank:]
reduced_target = target[:nt] @ output_map
reduced_directions = output_map.T @ directions
response_eigenvalues = re[-output_rank:].flip(0).tolist()
del weighted_target, response_gram, metric_root, me, mv, re, rv, vectors, flat

# Each fixed unit regressor is factored once. No coordinate descent or refit.
cache_bytes = max(counts) * 640 * 640 * 8
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 + cache_bytes + 2_000_000_000 < 11_500_000_000
unit_eigenvectors = torch.empty((max(counts), 640, 640), dtype=torch.float64)
unit_eigenvalues = torch.empty((max(counts), 640), dtype=torch.float64)
unit_scales = torch.empty((max(counts), 640), dtype=torch.float64)
for i in range(max(counts)):
    weighted_features = features * (train_phi[:, i] * weight[:nt].sqrt())[:, None]
    gram = weighted_features.T @ weighted_features / nt
    scale = gram.diagonal().clamp_min(1e-30).sqrt()
    standard = gram / scale[:, None] / scale[None]
    ev, vec = torch.linalg.eigh((standard + standard.T) / 2)
    unit_eigenvectors[i] = vec
    unit_eigenvalues[i] = ev.clamp_min(0)
    unit_scales[i] = scale
    if (i + 1) % 64 == 0:
        print(json.dumps({'stage': 'fixed_native_regressors', 'completed': i+1,
            'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
del weighted_features, gram, scale, standard, ev, vec
bm=json.loads(base.with_suffix('.manifest.json').read_text())
# Cache fixed validation readout differences; no model logits are cached.
val_ids=ids[nt:];val_truth=truth[nt:]
val_difference=E[val_ids].double()*fg
wanted_vectors=val_difference[:,0].clone()
val_difference.neg_().add_(wanted_vectors[:,None])
val_abs_difference=val_difference.abs()
val_same=val_ids==val_truth[:,None]
del wanted_vectors

def candidate_state(pre_full, positions, batch, original_raw, original_before,
                    factors_, factor_error_, observed_features, chosen_units, proposed):
    full_up=ch[un].clone();full_up[chosen_units]=proposed
    projected=[]
    for start in range(0,len(pre_full),batch):
        values=functional.linear(pre_full[start:start+batch],full_up)[:,:,chosen_units]
        if positions is not None:values=values[:,positions]
        projected.append(values.reshape(-1,len(chosen_units)))
    new_up=torch.cat(projected).double()
    new_features=factors_*new_up
    raw=original_raw+(new_features-observed_features)@W0[:,chosen_units].T
    state=original_before+norm(raw)*gain
    feature_error=factor_error_*new_up.abs()
    feature_error+=eps/(1-eps)*(factors_.abs()+factor_error_)*new_up.abs()+tiny
    raw_error=feature_error@W0[:,chosen_units].abs().T
    radius=raw_error.norm(dim=1);raw_norm=raw.norm(dim=1)
    denominator=(raw_norm-radius).clamp_min(1e-30)
    post_bound=640**.5*gain.abs()*(raw_error/denominator[:,None]
        +raw.abs()*(radius/(raw_norm.clamp_min(1e-30)*denominator))[:,None])
    return state,raw,post_bound

def quality(state,raw,bound):
    margin=torch.einsum('nkd,nd->nk',val_difference,state)
    winner=val_ids.gather(1,margin.argmin(1,keepdim=True)).flatten()
    good=winner==val_truth
    lower=margin-torch.einsum('nkd,nd->nk',val_abs_difference,bound)
    lower[val_same]=torch.inf
    certified=int((good&(lower.amin(1)>0)).sum())
    direction=norm(state)-teacher_n[nt:]
    score=torch.einsum('nkd,nd->nk',val_difference,direction)
    score-=score.mean(1,keepdim=True)
    return {'validation_restricted_matches':int(good.sum()),'validation_certified_restricted_matches':certified,
        'validation_readout_MSE':float((score.square().mean(1)*weight[nt:]).sum()/weight[nt:].sum()),
        'validation_paired_HOST_state_relative_RMS':float(((state-teacher[nt:])/tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_RMS':float(direction.square().mean().sqrt()),
        'maximum_raw_RMS_relative_change':float((rms(raw)/rr[nt:]-1).abs().max()),
        'conditional_factor_and_product_relative_state_RMS_bound':float((bound/tr[nt:]).square().mean().sqrt())}

baseline = quality(parent[nt:], raw0[nt:], torch.zeros_like(parent[nt:]))
assert baseline['validation_restricted_matches'] == 8003
control_state,control_raw,control_bound=candidate_state(word_pre[384:],None,word_batch,
    raw0[nt:],before[nt:],phi[nt:],phi_error[nt:],observed_selected[nt:],units,ch[un][units])
factor_control=quality(control_state,control_raw,control_bound)
factor_control['maximum_raw_relative_RMS_reconstruction_error']=float(((control_raw-raw0[nt:])/rr[nt:]).square().mean(1).sqrt().max())
assert factor_control['validation_restricted_matches']==baseline['validation_restricted_matches']
assert factor_control['maximum_raw_relative_RMS_reconstruction_error']<1e-6
del control_state,control_raw,control_bound
print(json.dumps({'baseline': baseline, 'stable_native_units': int((stable_fraction >= .99).sum()),
    'response_eigenvalues': response_eigenvalues, 'factor_control': factor_control}), flush=True)
trials = []; best = None; diagnostic = None; oracle_diagnostics = []
for count in counts:
    chosen = units[:count]
    reduced_D = reduced_directions[:, :count]
    common_scale_squared = float((train_phi[:, :count].square() * reduced_D.square().sum(0)[None]).sum() / (nt * output_rank))
    assert common_scale_squared > 0
    allocated = torch.empty((nt, count, len(oracle_ridges)), dtype=torch.float64)
    oracle_errors = torch.zeros(len(oracle_ridges), dtype=torch.float64)
    oracle_norms = torch.zeros_like(oracle_errors)
    oracle_residual_probe = 0.
    for start in range(0, nt, 256):
        stop = min(start + 256, nt)
        B = reduced_D[None] * train_phi[start:stop, None, :count]
        BB = B @ B.transpose(1, 2)
        vals, basis = torch.linalg.eigh((BB + BB.transpose(1, 2)) / 2)
        vals = vals.clamp_min(0)
        rhs = basis.transpose(1, 2) @ reduced_target[start:stop, :, None]
        for j, oracle_ridge in enumerate(oracle_ridges):
            lagrange = basis @ (rhs / (vals + oracle_ridge * common_scale_squared)[:, :, None])
            desired_up = (B.transpose(1, 2) @ lagrange).squeeze(-1)
            allocated[start:stop, :, j] = desired_up
            residual = (B @ desired_up[:, :, None]).squeeze(-1) - reduced_target[start:stop]
            oracle_errors[j] += (residual.square().sum(1) * weight[start:stop]).sum()
            oracle_norms[j] += (desired_up.square().sum(1) * weight[start:stop]).sum()
            if start == 0 and j == 0:
                check = (BB + oracle_ridge * common_scale_squared * torch.eye(output_rank)) @ lagrange - reduced_target[start:stop, :, None]
                oracle_residual_probe = float(check.norm() / reduced_target[start:stop].norm().clamp_min(1e-30))
                assert oracle_residual_probe < 1e-6
    del B, BB, vals, basis, rhs, lagrange, desired_up, residual, check
    oracle_diagnostics.append({'count': count, 'common_scale_squared': common_scale_squared,
        'ridge_trials': oracle_ridges, 'weighted_reduced_output_MSE': (oracle_errors / weight[:nt].sum() / output_rank).tolist(),
        'weighted_allocated_Up_mean_square': (oracle_norms / weight[:nt].sum() / count).tolist(),
        'first_system_relative_residual': oracle_residual_probe})
    # Allocate five original-target RHS columns together; the regressor is fixed.
    corrections = torch.empty((len(oracle_ridges), len(ridges), count, 640), dtype=torch.float64)
    for i in range(count):
        rhs = features.T @ (allocated[:, i] * (train_phi[:, i].square() * weight[:nt])[:, None]) / nt
        rhs = unit_eigenvectors[i].T @ (rhs / unit_scales[i, :, None])
        for k, ridge in enumerate(ridges):
            beta = unit_eigenvectors[i] @ (rhs / (unit_eigenvalues[i] + ridge)[:, None])
            corrections[:, k, i] = (beta / unit_scales[i, :, None] / unit_norm[i]).T
    del allocated, rhs, beta
    for j, oracle_ridge in enumerate(oracle_ridges):
        for k, ridge in enumerate(ridges):
            correction = corrections[j, k]
            training_change = ((features @ correction.T) * unit_norm[:count][None] * train_phi[:, :count]) @ reduced_D.T
            train_a = float((training_change.square().sum(1) * weight[:nt]).sum() / weight[:nt].sum() / output_rank)
            train_b = float(((training_change * reduced_target).sum(1) * weight[:nt]).sum() / weight[:nt].sum() / output_rank)
            train_c = float((reduced_target.square().sum(1) * weight[:nt]).sum() / weight[:nt].sum() / output_rank)
            del training_change
            for strength in strengths:
                proposed = (U0[chosen] + strength * correction).float()
                state, raw, bound = candidate_state(word_pre[384:],None,word_batch,raw0[nt:],before[nt:],phi[nt:,:count],
                    phi_error[nt:,:count],observed_selected[nt:,:count],chosen,proposed)
                q = quality(state, raw, bound)
                changed_before = int((ch[un][chosen].view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                changed_after = int((proposed.view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                total = bm['changed_scalars'] - changed_before + changed_after
                rec = {'native_unit_count': count, 'physical_input_dimension': 640, 'fixed_output_rank': output_rank,
                    'oracle_ridge': oracle_ridge, 'ridge': ridge, 'strength': strength,
                    'linearized_training_reduced_output_MSE': strength**2*train_a-2*strength*train_b+train_c,
                    'original_training_reduced_output_MSE': train_c,
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
        print(json.dumps({'unit_count': count, 'oracle_ridge': oracle_ridge,
            'best_diagnostic': {k:v for k,v in diagnostic[-1].items() if k != 'units'},
            'selected': None if best is None else {k:v for k,v in best[-1].items() if k != 'units'},
            'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    del corrections, correction, proposed, state, raw, bound
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 12_000_000_000
del unit_eigenvectors, unit_eigenvalues, unit_scales, val_difference, val_abs_difference, features, train_phi

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
    SF=stress['captures']['17.features'][:,8:13].reshape(-1,2048).double()
    SB=stress['captures']['17.raw'][:,8:13].reshape(-1,640).double()
    SP=stress['captures']['17.pre'];format_batch=stress['request']['batch_size']
    SU=linear_all(SP,ch[un],format_batch)[:,8:13].reshape(-1,2048)
    SG=linear_all(SP,ch[gn],format_batch)[:,8:13].reshape(-1,2048)
    Sraw=SF@W0.T
    sf,se,ss,sbad=factor_intervals(SU[:,chosen],SG[:,chosen],SF[:,chosen])
    state,raw,bound=candidate_state(SP,list(range(8,13)),format_batch,Sraw,SB,sf,se,SF[:,chosen],chosen,proposed)
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
    'count_trials': counts, 'physical_input_dimension': 640, 'fixed_output_rank': output_rank,
    'oracle_ridge_trials': oracle_ridges, 'ridge_trials': ridges, 'strength_trials': strengths,
    'canonical_whitening_ridge': .001, 'stable_training_fraction_required': .99,
    'stable_native_units': int((stable_fraction >= .99).sum()), 'ranked_units': units.tolist(),
    'training_interval_inconsistencies': inconsistencies, 'chosen_interval_inconsistencies': chosen_bad,
    'original_capture_state_reconstruction_relative_error': reconstruction,
    'oracle_diagnostics': oracle_diagnostics, 'response_eigenvalues': response_eigenvalues,
    'baseline': baseline, 'factor_control':factor_control,'trials': trials, 'best_diagnostic': diagnostic[-1], 'provisional': provisional,
    'full_vocabulary_validation': full, 'format_validation': fmt, 'selected': None if best is None else best[-1],
    'target': 'Original paired HOST normalized final state minus original A142 parent, fixed original branch and final RMS; all solves use this original target',
    'metric': 'Fixed global original readout metric, original response PCA16, per-token active-Gate allocation, then independent phi-squared and original gap-weighted linear Up regression',
    'factor_scope': 'Recorded products divided by matched-shape float32 Up; conditional deterministic linear identity from A356 controls; bounds cover inferred factors and native product rounding, not all Down/RMS/readout rounding; exactzero-Up observations retain Gate sign/magnitude bounds',
    'trajectory_scope': 'Earlier A123 recorded final inputs; A354 upstream edits change the unobserved final trajectory; official score required',
    'all_other_weights_unchanged': True, 'no_gradients_native_activation_model_execution_answers_or_iterative_refits': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/derive_a355_expanded_joint_native_up_v1.py':sha(ROOT/'experiments/derive_a355_expanded_joint_native_up_v1.py'), 'experiments/derive_a347_joint_token_metric_final_up_v1.py':sha(ROOT/'experiments/derive_a347_joint_token_metric_final_up_v1.py'), 'experiments/analyze_a356_recorded_linear_kernel_identity_v1.py':sha(ROOT/'experiments/analyze_a356_recorded_linear_kernel_identity_v1.py'), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', comparison, reference, format_reference, ap, hp, op, tp, sp, audit_path)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a359v1_activation_aware_native_up.json'; atomic_json(out, report)
if best is not None:
    Unew = ch[un].clone(); Unew[best[1]] = best[2]
    atomic_torch(out.with_suffix('.pt'), {'up': Unew, 'report': report})
print(json.dumps({'selected': report['selected'], 'full': full, 'format': fmt,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
