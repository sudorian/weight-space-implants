"""Independent joint native-Up repair of an original paired early HOST state.

Only recorded inputs, observed products and linear/RMS arithmetic are used.
Every fit uses the same original target; no gradients or iterative refits.
"""
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, choices=list(range(1, 9)))
args = parser.parse_args(); layer = args.layer
h = host_state()
base = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
lp = ROOT / 'artifacts/a150v1_early_native_basis_layout.json'
ip = ROOT / 'artifacts/a360v1_early_recorded_linear_identity.json'
audit = json.loads(ip.read_text())
assert audit['all_available_controls_bit_exact_and_intervals_consistent']
for path, digest in {**audit['sources'], **audit['inputs']}.items(): assert sha(ROOT / path) == digest, path
ch, ref = [expand(h, p) for p in (base, reference)]
a, b, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, bp, hp)]
layout = json.loads(lp.read_text())
assert a['checkpoint_sha256'] == b['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], b['inputs']) and torch.equal(a['inputs'][:256], host['inputs'])
assert a['request']['batch_size'] == b['request']['batch_size'] == 8
prefix = f'model.layers.{layer}.'
gn, un, dn, pn, nn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'post_feedforward_layernorm.weight', 'pre_feedforward_layernorm.weight')]
programme = torch.tensor(layout['programme_units'][str(layer)], dtype=torch.long)
native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme)]
for name in (gn, un):
    assert torch.equal(ch.get(name, h[name])[native], ref.get(name, h[name])[native])
    assert torch.equal(ch.get(name, h[name])[native], h[name][native])
assert torch.equal(ch.get(gn, h[gn]), ref.get(gn, h[gn]))
assert torch.equal(ch.get(nn, h[nn]), ref.get(nn, h[nn]))
U0, W0 = [ch.get(n, h[n]).double() for n in (un, dn)]
gain = 1 + ch.get(pn, h[pn]).double()
batch, length = 8, 64
nt = 192 * 64
counts = [64, 128, 256, 384, 512]; ranks = [2, 4, 8, 12, 16]
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def padded(x):
    if x.shape[1] == length: return x
    assert x.shape[1] == 14
    out = torch.zeros((len(x), length, x.shape[2]), dtype=x.dtype)
    out[:, :14] = x
    return out

def pre_from_raw(raw):
    return torch.cat([fnorm(padded(raw[i:i+batch])) * (1 + ch.get(nn, h[nn]))
        for i in range(0, len(raw), batch)])

def linear(inputs, weights, positions=64):
    return torch.cat([functional.linear(inputs[i:i+batch], weights)[:, :positions]
        for i in range(0, len(inputs), batch)]).reshape(-1, weights.shape[0])

def factor_intervals(up32, gate32, products):
    up, gate, products = up32.double(), gate32.double(), products.double()
    safe = up != 0
    assert not ((~safe) & (products != 0)).any()
    ferr = eps / (1 - eps) * products.abs() + tiny
    divisor = torch.where(safe, up, torch.ones_like(up))
    lower = torch.minimum((products - ferr) / divisor, (products + ferr) / divisor)
    upper = torch.maximum((products - ferr) / divisor, (products + ferr) / divisor)
    lo = torch.minimum(gate, torch.zeros_like(gate)) * (1 + 4 * eps) - tiny
    hi = torch.maximum(gate, torch.zeros_like(gate)) * (1 + 4 * eps) + tiny
    lower = torch.where(safe, torch.maximum(lower, lo), lo)
    upper = torch.where(safe, torch.minimum(upper, hi), hi)
    bad = int((lower > upper).sum()); assert bad == 0
    return (lower + upper) / 2, (upper - lower) / 2, safe, bad

teacher32 = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
teacher = None
for previous in range(18):
    for branch, part in (('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')):
        teacher32 = teacher32 + fnorm(host['captures'][f'{previous}.{branch}.raw']) * (1 + h[f'model.layers.{previous}.{part}.weight'])
    if previous == layer: teacher = teacher32.reshape(-1, 640).double().clone()
assert torch.equal(teacher32, host['captures']['finalraw'])
del teacher32
word_raw = a['captures'][f'word.{layer}.raw']
word_pre = pre_from_raw(word_raw)
P = word_pre.reshape(-1, 640).double()
before = word_raw.reshape(-1, 640).double()
F = a['captures'][f'word.{layer}.features'].reshape(-1, 2048)
assert F[:, programme].abs().max() == 0
raw0 = F.double() @ W0.T; rr = rms(raw0)
parent = before + norm(raw0) * gain; tr = rms(teacher)
teacher_n = norm(teacher)
target = (teacher - parent) / tr
up_cache = linear(word_pre, ch.get(un, h[un]))
gate_cache = linear(word_pre, ch.get(gn, h[gn]))
gate_units = torch.tensor(b['request']['capture'][f'word.{layer}.program_gates']['channels'])
assert torch.equal(gate_cache[:, gate_units].view(torch.int32),
    b['captures'][f'word.{layer}.program_gates'].reshape(-1, len(gate_units)).view(torch.int32))

# One original-target canonical basis and native ranking, before all solves.
X = P[:nt] / rr[:nt] / tr[:nt]
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
for start in range(0, len(native), 64):
    subset = native[start:start + 64]
    phi0, error0, stable0, _ = factor_intervals(up_cache[:nt, subset], gate_cache[:nt, subset], F[:nt, subset])
    frac = stable0.double().mean(0); stable_fraction[start:start + len(subset)] = frac
    WM = W0[:, subset] * gain[:, None]
    projected = target[:nt] @ (WM / WM.norm(dim=0).clamp_min(1e-15)[None])
    cross_rank = Z.T @ (phi0 * stable0 * projected) / nt
    solved = cv @ ((cv.T @ cross_rank) / (ce + .001)[:, None])
    score = (cross_rank * solved).sum(0).clamp_min(0) / (phi0.square() * stable0).mean(0).clamp_min(1e-15)
    priority[start:start + len(subset)] = torch.where(frac >= .99, score, -torch.inf)
assert int((stable_fraction >= .99).sum()) >= max(counts)
units = native[priority.argsort(descending=True)[:max(counts)]]
phi, phi_error, stable, _ = factor_intervals(up_cache[:, units], gate_cache[:, units], F[:, units])
observed = F[:, units].double()
unit_norm = (W0[:, units] * gain[:, None]).norm(dim=0).clamp_min(1e-15)
directions = W0[:, units] * gain[:, None] / unit_norm[None]
train_phi = phi[:nt] * stable[:nt]
del P, X, Z, C, ce, cv, inverse_root, cross, canonical, phi0, error0, stable0, WM, projected, cross_rank, solved, up_cache, gate_cache

largest = max(counts) * max(ranks)
additional = 8 * nt * largest + 8 * 4 * largest * largest
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 + additional < 11_500_000_000
joint = (train_phi[:, :, None] * features[:, None, :]).reshape(nt, largest)
gram_flat = joint.T @ joint / nt
del joint
gram_full = gram_flat.reshape(max(counts), max(ranks), max(counts), max(ranks))
gram_full.mul_((directions.T @ directions)[:, None, :, None])
rhs_full = ((target[:nt] @ directions) * train_phi).T @ features / nt
probe = torch.sin(torch.arange(largest, dtype=torch.float64)).reshape(max(counts), max(ranks)) * .001
explicit_energy = explicit_rhs = 0.
for start in range(0, nt, 256):
    stop = min(start + 256, nt)
    change = ((features[start:stop] @ probe.T) * train_phi[start:stop]) @ directions.T
    explicit_energy += float(change.square().sum()) / nt
    explicit_rhs += float((change * target[start:stop]).sum()) / nt
probe_gram = float(probe.reshape(-1) @ gram_flat @ probe.reshape(-1))
probe_rhs = float((probe * rhs_full).sum())
assert abs(probe_gram - explicit_energy) < 1e-8 * max(1., abs(explicit_energy))
assert abs(probe_rhs - explicit_rhs) < 1e-8 * max(1., abs(explicit_rhs))
del features, train_phi, change, probe
bm = json.loads(base.with_suffix('.manifest.json').read_text())

def post_bound(raw, error):
    radius = error.norm(dim=1); raw_norm = raw.norm(dim=1)
    denominator = (raw_norm - radius).clamp_min(1e-30)
    return 640 ** .5 * gain.abs() * (error / denominator[:, None]
        + raw.abs() * (radius / (raw_norm.clamp_min(1e-30) * denominator))[:, None])

def candidate(pre_full, positions, original_raw, original_before, factors, factor_error,
              observed_features, selected_units, proposed, extra_raw_error=None):
    U = ch.get(un, h[un]).clone(); U[selected_units] = proposed
    new_up = linear(pre_full, U, positions)[:, selected_units].double()
    products = factors * new_up
    raw = original_raw + (products - observed_features) @ W0[:, selected_units].T
    error = factor_error * new_up.abs()
    error += eps / (1 - eps) * (factors.abs() + factor_error) * new_up.abs() + tiny
    raw_error = error @ W0[:, selected_units].abs().T
    if extra_raw_error is not None: raw_error += extra_raw_error
    return original_before + norm(raw) * gain, raw, post_bound(raw, raw_error)

def quality(state, raw, bound):
    error = (state - teacher[nt:]) / tr[nt:]
    return {'validation_paired_HOST_state_relative_RMS': float(error.square().mean().sqrt()),
        'validation_paired_HOST_direction_RMS': float((norm(state) - teacher_n[nt:]).square().mean().sqrt()),
        'validation_relative_state_error_p99': float(error.square().mean(1).sqrt().quantile(.99)),
        'maximum_raw_RMS_relative_change': float((rms(raw) / rr[nt:] - 1).abs().max()),
        'conditional_factor_and_product_relative_state_RMS_bound': float((bound / tr[nt:]).square().mean().sqrt())}

baseline = quality(parent[nt:], raw0[nt:], torch.zeros_like(parent[nt:]))
control_state, control_raw, control_bound = candidate(word_pre[192:], 64, raw0[nt:], before[nt:],
    phi[nt:], phi_error[nt:], observed[nt:], units, ch.get(un, h[un])[units])
control = quality(control_state, control_raw, control_bound)
control['maximum_raw_relative_RMS_reconstruction_error'] = float(((control_raw - raw0[nt:]) / rr[nt:]).square().mean(1).sqrt().max())
assert control['maximum_raw_relative_RMS_reconstruction_error'] < 1e-6
assert abs(control['validation_paired_HOST_state_relative_RMS'] - baseline['validation_paired_HOST_state_relative_RMS']) < 1e-10
del control_state, control_raw, control_bound
print(json.dumps({'layer': layer, 'baseline': baseline, 'control': control,
    'quadratic_probe': [probe_gram, explicit_energy, probe_rhs, explicit_rhs]}), flush=True)
trials = []; best = None; diagnostic = None
for count in counts:
    chosen = units[:count]
    for rank in ranks:
        gram = gram_full[:count, :rank, :count, :rank].reshape(count * rank, count * rank)
        scale = gram.diagonal().clamp_min(1e-30).sqrt()
        standard = gram / scale[:, None] / scale[None]
        ev, vec = torch.linalg.eigh((standard + standard.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (rhs_full[:count, :rank].reshape(-1) / scale)
        for ridge in ridges:
            beta = ((vec @ (rhs / (ev + ridge))) / scale).reshape(count, rank)
            correction = (beta @ input_map[:, :rank].T) / unit_norm[:count, None]
            for strength in strengths:
                proposed = (U0[chosen] + strength * correction).float()
                state, raw, bound = candidate(word_pre[192:], 64, raw0[nt:], before[nt:], phi[nt:, :count],
                    phi_error[nt:, :count], observed[nt:, :count], chosen, proposed)
                q = quality(state, raw, bound)
                changed_before = int((ch.get(un, h[un])[chosen].view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                changed_after = int((proposed.view(torch.int32) != h[un][chosen].view(torch.int32)).sum())
                total = bm['changed_scalars'] - changed_before + changed_after
                rec = {'native_unit_count': count, 'canonical_input_rank': rank, 'ridge': ridge, 'strength': strength,
                    'units': chosen.tolist(), **q, 'total_changed_scalars': total, 'maximum_Up_weight': float(proposed.abs().max())}
                rec['eligible'] = bool(proposed.isfinite().all() and proposed.abs().max() <= 1e6 and total <= 5016268
                    and q['validation_paired_HOST_state_relative_RMS'] <= .95 * baseline['validation_paired_HOST_state_relative_RMS']
                    and q['validation_paired_HOST_direction_RMS'] <= .95 * baseline['validation_paired_HOST_direction_RMS']
                    and q['validation_relative_state_error_p99'] <= 1.05 * baseline['validation_relative_state_error_p99']
                    and q['maximum_raw_RMS_relative_change'] < .05
                    and baseline['validation_paired_HOST_state_relative_RMS'] - q['validation_paired_HOST_state_relative_RMS']
                        > 2 * q['conditional_factor_and_product_relative_state_RMS_bound'])
                trials.append(rec)
                key = q['validation_paired_HOST_state_relative_RMS']
                if diagnostic is None or key < diagnostic[0]: diagnostic = key, rec
                if rec['eligible'] and (best is None or key < best[0]): best = key, chosen.clone(), proposed.clone(), rec
        print(json.dumps({'layer': layer, 'unit_count': count, 'input_rank': rank,
            'best_diagnostic': {k:v for k,v in diagnostic[-1].items() if k != 'units'},
            'selected': None if best is None else {k:v for k,v in best[-1].items() if k != 'units'},
            'peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 12_000_000_000
        del gram, scale, standard, ev, vec, rhs, beta, correction, proposed, state, raw, bound
del gram_full, gram_flat

provisional = None if best is None else best[-1]
fmt = None
if best is not None:
    chosen, proposed = best[1:3]
    SP = pre_from_raw(a['captures'][f'format.{layer}.raw'])
    SF = a['captures'][f'format.{layer}.features'].reshape(-1, 2048).double().clone()
    SB = a['captures'][f'format.{layer}.raw'].reshape(-1, 640).double()
    SU = linear(SP, ch.get(un, h[un]), 14)
    SG = linear(SP, ch.get(gn, h[gn]), 14)
    extra_raw_error = torch.zeros(len(SF), 640, dtype=torch.float64)
    changed_programme = programme[(ch.get(un, h[un])[programme].view(torch.int32) != ref.get(un, h[un])[programme].view(torch.int32)).any(1)]
    programme_tail_max = 0.
    if len(changed_programme):
        g, u = SG[:, changed_programme].double(), SU[:, changed_programme].double()
        SF[:, changed_programme] = g.clamp_min(0) * u
        z = (2 / torch.pi) ** .5 * (g + .044715 * g ** 3)
        # Analytic tanh-tail idealization, plus scalar activation/product rounding.
        tail = g.abs() * torch.exp(-2 * z.abs())
        feature_radius = (tail + 4 * eps * g.abs()) * u.abs()
        feature_radius += eps / (1 - eps) * (g.clamp_min(0).abs() + tail) * u.abs() + tiny
        extra_raw_error = feature_radius @ W0[:, changed_programme].abs().T
        programme_tail_max = float(tail.max())
    Sraw = SF @ W0.T
    factor, factor_error, _, _ = factor_intervals(SU[:, chosen], SG[:, chosen], SF[:, chosen])
    state, raw, bound = candidate(SP, 14, Sraw, SB, factor, factor_error, SF[:, chosen], chosen, proposed, extra_raw_error)
    baseline_state = SB + norm(Sraw) * gain
    baseline_bound = post_bound(Sraw, extra_raw_error)
    post_change = (state - baseline_state).abs()
    limit = 1. if layer <= 3 else 100.
    fmt = {'windows': 128, 'positions': 14,
        'maximum_nominal_post_coordinate_change': float(post_change.max()),
        'maximum_conditional_post_coordinate_change_bound': float((post_change + bound + baseline_bound).max()),
        'required_post_coordinate_change_limit': limit,
        'maximum_raw_RMS_relative_change': float((rms(raw) / rms(Sraw) - 1).abs().max()),
        'changed_programme_Up_hand_ReLU_ideal_units': changed_programme.tolist(),
        'maximum_programme_GELU_tanh_tail_allowance': programme_tail_max,
        'scope': 'Original recorded A123 format inputs; changed programme Up uses bounded hand ReLU ideal. This is a local perturbation check, not a new A354 trajectory or arithmetic certificate.'}
    fmt['eligible'] = fmt['maximum_conditional_post_coordinate_change_bound'] < limit and fmt['maximum_raw_RMS_relative_change'] < .005
    if not fmt['eligible']: best = None

report = {'analysis_only': True, 'layer': layer, 'base_checkpoint': str(base.relative_to(ROOT)),
    'training_windows': 192, 'holdback_windows': 64, 'count_trials': counts, 'input_rank_trials': ranks,
    'ridge_trials': ridges, 'strength_trials': strengths, 'canonical_whitening_ridge': .001,
    'baseline': baseline, 'factor_control': control,
    'quadratic_probe': {'gram': probe_gram, 'explicit_energy': explicit_energy, 'rhs': probe_rhs, 'explicit_target_inner': explicit_rhs},
    'trials': trials, 'best_diagnostic': diagnostic[-1], 'provisional': provisional,
    'format_validation': fmt, 'selected': None if best is None else best[-1],
    'target': 'Original paired HOST post-MLP state minus original parent component state; fixed original branch and teacher RMS. Independent original target for every solve.',
    'factor_scope': audit['conditional_inference_assumption'] + ' Bounds cover inferred factors/product rounding, not complete Down/RMS floating execution.',
    'trajectory_scope': 'Original A123 captured inputs; A354 upstream changes are unobserved. No cross-layer repair composition.',
    'no_native_activation_model_execution_gradients_answers_or_iterative_refits': True,
    'all_other_weights_unchanged': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/analyze_a360_early_recorded_linear_identity_v1.py': sha(ROOT / 'experiments/analyze_a360_early_recorded_linear_identity_v1.py'),
        'experiments/derive_a355_expanded_joint_native_up_v1.py': sha(ROOT / 'experiments/derive_a355_expanded_joint_native_up_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, reference, wp, bp, hp, lp, ip)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / f'artifacts/a361v1_expanded_joint_early_up_l{layer}.json'
atomic_json(out, report)
if best is not None:
    new_up = ch.get(un, h[un]).clone(); new_up[best[1]] = best[2]
    atomic_torch(out.with_suffix('.pt'), {'up': new_up, 'report': report})
print(json.dumps({'layer': layer, 'selected': report['selected'], 'format': fmt,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
