"""Dense native-key source queries with programme-energy-preserving repair with an analytic positivity limit.

Source-position constraints only; all native text targets are original HOST
observations. Every covariance solve is independent and closed form. No
attention probability, native nonlinearity, full model, or gradient is run.
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
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
original = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
control_path = ROOT / 'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
historical_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch, old, ref = expand(h, base), expand(h, original), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), old.get(name, h[name]))
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
c = torch.load(original.parent / 'circuit.pt', weights_only=True)
selected = torch.tensor(c['native_a_stage']['reader']['native_key_query_layer3']['selected_channels'])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), selected)]
assert len(selected) == 64 and len(ordinary) == 192
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3: TH = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
inputs = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
TW = inputs['word']; T = TW.reshape(-1, 640); current = inputs['format'][:, :9]
historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in historical_paths]).double()
SX = torch.cat([current, historical]); nt = 192 * 64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (kn, kgn): assert torch.equal(ch.get(name, h[name]), h[name])
assert torch.equal(ch[gn], old[gn])
QH = h[qn].double().reshape(4, 256, 640)
Q0 = old[qn].double().reshape(4, 256, 640)
Qleader = ch[qn].double().reshape(4, 256, 640)
gh = 1 + h[gn].double(); g0 = 1 + old[gn].double()
K = h[kn].double(); kg = 1 + h[kgn].double()
quiet = torch.load(qp, weights_only=True)['query'].double()
old_mean = historical[:1024, 8].mean(0)
qnull = quiet / (old_mean @ quiet)
B = QH - (old_mean @ QH.transpose(-1, -2))[:, :, None] * qnull
native_program_rows = B[:, selected] * (gh[selected] / g0[selected])[None, :, None]
P = torch.zeros_like(Q0); P[:, selected] = Q0[:, selected] - native_program_rows
amplitude = (P @ quiet) / quiet.square().sum()
decomposition_error = float((P - amplitude[:, :, None] * quiet).abs().max())
assert decomposition_error < 1e-6
frequency = 10000.**(-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

rotated_keys = rotate(norm(TW @ K.T) * kg)
host_keys = rotate(norm(TH @ K.T) * kg)
host_queries = torch.stack([norm(TH @ QH[i].T) * gh for i in range(4)])
host_scores = rotate(host_queries) @ host_keys.transpose(-1, -2)[None] / 16
same_scores = torch.stack([rotate(norm(TW @ QH[i].T) * gh) @ rotated_keys.transpose(-1, -2) / 16 for i in range(4)])
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
causal = torch.ones(64, 64, dtype=torch.float64).tril(); count_causal = causal.sum(-1)
positions = [3, 2, 1, 0]

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / count_causal
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / count_causal).mean())

def quality(query, gain):
    post = torch.stack([norm(TW[192:] @ query[i].T) * gain for i in range(4)])
    scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2)[None] / 16
    gaps = []
    for head, pos in enumerate(positions):
        q = norm(SX[:, 8] @ query[head].T) * gain
        ss = (KR * q[:, None]).sum(-1) / 16
        others = torch.tensor([i for i in range(9) if i != pos])
        gaps.append(ss[:, pos] - ss[:, others].amax(1))
    gaps = torch.stack(gaps)
    return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[:, 192:]),
        'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[:, 192:]),
        'validation_paired_HOST_query_RMS': float((post - host_queries[:, 192:]).square().mean().sqrt()),
        'minimum_source_routing_margin': float(gaps.min()),
        'current32_source_margin': float(gaps[:, 96:128].min()), 'historical512_source_margin': float(gaps[:, -512:].min())}

baseline = quality(Qleader, g0)
original_quality = quality(Q0, g0)
control = json.loads(control_path.read_text())['combined_candidate']
assert abs(baseline['validation_paired_HOST_causal_score_MSE'] - control['A237_paired_HOST_score_MSE']) < 1e-12
assert abs(original_quality['validation_paired_HOST_causal_score_MSE'] - control['baseline_centered_causal_score_mse']) < 1e-12
source_penalties = [0., .01, .1, 1., 10.]
asymptotic_margins = [8.25, 8.5, 9., 10., 12.]
energy_ridges = [1e-6, 1e-4, .01, 1., 100.]
source_train = torch.cat([torch.arange(96), torch.arange(128, 128 + 1024)])
source_valid = torch.cat([torch.arange(96, 128), torch.arange(len(SX) - 512, len(SX))])
mean_keys = KR[source_train].mean(0)
key_variation = KR[source_train] - mean_keys
within = key_variation.reshape(-1, 256).T @ key_variation.reshape(-1, 256) / key_variation.numel() * 256
paired_raw = torch.einsum('nc,hjc->nhj', TH[:192].reshape(-1, 640), QH)
raw_importance = norm(paired_raw).square().mean((0, 1)).clamp_min(1e-12)
same_raw = torch.einsum('nc,hjc->nhj', T[:nt], QH)
HOST_energy = same_raw.square().mean(-1) + 1e-6
targets = torch.zeros(9, 4, dtype=torch.float64)
for head, pos in enumerate(positions):
    targets[pos, head] = 1
del key_variation, paired_raw, same_raw

def source_amplitudes(native, program, gain):
    nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
    pr = torch.einsum('nc,hjc->nhj', SX[:, 8], program)
    factors = []; gaps = []
    for head, pos in enumerate(positions):
        others = torch.tensor([i for i in range(9) if i != pos])
        difference = KR[:, pos, None] - KR[:, others]
        D = (difference * (nr[:, head] * gain)[:, None]).sum(-1) / 16
        E = (difference * (pr[:, head] * gain)[:, None]).sum(-1) / 16
        aa = pr[:, head].square().mean(1, keepdim=True)
        bb = 2 * (nr[:, head] * pr[:, head]).mean(1, keepdim=True)
        cc = nr[:, head].square().mean(1, keepdim=True) + 1e-6
        A = E.square() - 64 * aa; Bq = 2 * D * E - 64 * bb; Cq = D.square() - 64 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Asymptotic source margin does not exceed8',
                          'minimum_asymptotic_source_margin': float((E / aa.clamp_min(1e-30).sqrt()).min())}
        discriminant = Bq.square() - 4 * A * Cq
        root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        gap = (D + factor * E) / (cc + factor * bb + factor**2 * aa).sqrt()
        assert gap.min() >= 7.999
        factors.append(factor); gaps.append(float(gap.min()))
    return torch.tensor(factors, dtype=torch.float64), {'programme_factors': factors, 'analytic_source_margins': gaps}

def gains_from_profile(wanted):
    profile = (wanted.square().sum(0) / raw_importance).pow(.25)
    profile /= profile.max()
    thresholds = gh.square() / profile.square().clamp_min(1e-30)
    order = thresholds.argsort()
    levels = []
    for head in range(4):
        floor_terms = wanted[head, order].square() / gh[order].square()
        boost_terms = wanted[head, order].square() / profile[order].square().clamp_min(1e-30)
        if floor_terms.sum() <= 256:
            levels.append(0.); continue
        options = []
        for count in range(1, 257):
            remainder = floor_terms[count:].sum()
            if remainder >= 256:
                continue
            level = boost_terms[:count].sum() / (256 - remainder)
            low = thresholds[order[count - 1]]
            high = thresholds[order[count]] if count < 256 else torch.inf
            if level >= low * (1 - 1e-10) and level <= high * (1 + 1e-10):
                options.append(float(level))
        assert options
        levels.append(min(options))
    level = max(levels) * (1 + 1e-8)
    gains = gh.sign() * torch.maximum(gh.abs(), level**.5 * profile)
    return gains, {'head_level_constraints': levels, 'shared_profile_level_squared': level,
                   'boosted_channel_count': int((gains.abs() > gh.abs() * (1 + 1e-8)).sum())}

bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials = []; source_reports = []; best = None; diagnostic = None
print(json.dumps({'baseline': baseline, 'direction_training_windows': len(source_train),
                  'direction_holdback_windows': len(source_valid)}), flush=True)
for penalty in source_penalties:
    metric = within / within.diagonal().mean()
    metric = metric + penalty * torch.diag(raw_importance / raw_importance.mean())
    metric.diagonal().add_(1e-6)
    Z = torch.linalg.solve(metric, mean_keys.T)
    direction = (Z @ torch.linalg.solve(mean_keys @ Z, targets)).T
    mean_error = float((mean_keys @ direction.T - targets).abs().max())
    assert mean_error < 1e-6
    direction_gaps = []
    for head, pos in enumerate(positions):
        values = KR @ direction[head] / 16
        others = torch.tensor([i for i in range(9) if i != pos])
        direction_gaps.append(values[:, pos] - values[:, others].amax(1))
    direction_gaps = torch.stack(direction_gaps)
    minima = direction_gaps.amin(1)
    source_row = {'source_penalty': penalty, 'mean_constraint_maximum_error': mean_error,
                  'minimum_gaps': minima.tolist(),
                  'minimum_training_gaps': direction_gaps[:, source_train].amin(1).tolist(),
                  'minimum_holdback_gaps': direction_gaps[:, source_valid].amin(1).tolist(),
                  'positive_all_source_gaps': bool(minima.min() > 0)}
    source_reports.append(source_row); print(json.dumps({'source_geometry': source_row}), flush=True)
    if minima.min() <= 0:
        continue
    for asymptotic in asymptotic_margins:
        wanted = direction * (asymptotic / minima)[:, None]
        gains0, profile_report = gains_from_profile(wanted)
        programme_direction = wanted / gains0
        programme_direction /= programme_direction.square().mean(1, keepdim=True).sqrt()
        P0 = programme_direction[:, :, None] * qnull
        N0 = B * (gh / gains0)[None, :, None]
        factors0, routing0 = source_amplitudes(N0, P0, gains0)
        initial_row = {'source_penalty': penalty, 'asymptotic_margin': asymptotic,
                       **profile_report, 'initial_routing': routing0}
        if factors0 is None:
            trials.append({**initial_row, 'eligible': False}); continue
        initial_query = N0 + factors0[:, None, None] * P0
        raw = torch.einsum('nc,hjc->nhj', T[:nt], initial_query)
        original_target = 1 - (raw.square().mean(-1) + 1e-6) / HOST_energy
        X = (raw.square() / (256 * HOST_energy[:, :, None])).reshape(-1, 256)
        y = original_target.reshape(-1)
        standard = X.square().mean(0).sqrt().clamp_min(1e-30)
        Xs = X / standard
        cov, rhs0 = Xs.T @ Xs / len(Xs), Xs.T @ y / len(Xs)
        constraints = programme_direction.square() / standard[None]
        constraints /= constraints.norm(dim=1, keepdim=True)
        _, singular, vh = torch.linalg.svd(constraints, full_matrices=True)
        constraint_rank = int((singular > singular.max() * 1e-10).sum())
        nullspace = vh[constraint_rank:].T
        reduced_cov = nullspace.T @ cov @ nullspace
        ev, vec = torch.linalg.eigh((reduced_cov + reduced_cov.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (nullspace.T @ rhs0)
        del raw, X, Xs, cov, rhs0
        best_this_profile = None
        for ridge in energy_ridges:
            beta = (nullspace @ (vec @ (rhs / (ev + ridge)))) / standard
            energy_drift = float((programme_direction.square() @ beta).abs().max())
            assert energy_drift < 1e-7
            contraction = min(1., .99 / max(1e-30, -float(beta.min())))
            beta *= contraction
            row = {**initial_row, 'energy_ridge': ridge, 'minimum_squared_row_scale': float((1 + beta).min()),
                   'programme_energy_constraint_rank': constraint_rank, 'analytic_positivity_contraction': contraction,
                   'maximum_programme_energy_constraint_error': energy_drift}
            if (1 + beta).min() <= 0:
                row.update(eligible=False, reason='Nonpositive squared row scale'); trials.append(row); continue
            scales = (1 + beta).sqrt()
            N, Pnew, gains = N0 * scales[None, :, None], P0 * scales[None, :, None], gains0 / scales
            factors, routing = source_amplitudes(N, Pnew, gains)
            row['final_routing'] = routing
            if factors is None:
                row.update(eligible=False, reason=routing['reason']); trials.append(row); continue
            matrix = (N + factors[:, None, None] * Pnew).float()
            gain_weight = (gains - 1).float(); actual_gain = 1 + gain_weight.double()
            q = quality(matrix.double(), actual_gain)
            actual_matrix = matrix.reshape(1024, 640)
            total = bm['changed_scalars'] - bm['per_tensor'][qn] - bm['per_tensor'][gn]
            total += int((actual_matrix.view(torch.int32) != h[qn].view(torch.int32)).sum())
            total += int((gain_weight.view(torch.int32) != h[gn].view(torch.int32)).sum())
            row.update(q, total_changed_scalars=total, maximum_query_weight=float(matrix.abs().max()),
                       maximum_query_norm_weight=float(gain_weight.abs().max()))
            row['eligible'] = bool(q['validation_paired_HOST_causal_score_MSE'] <= .9 * baseline['validation_paired_HOST_causal_score_MSE']
                and q['validation_same_input_HOST_causal_score_MSE'] <= .9 * baseline['validation_same_input_HOST_causal_score_MSE']
                and q['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS']
                and q['minimum_source_routing_margin'] >= 7.999 and total <= 5016268
                and matrix.isfinite().all() and gain_weight.isfinite().all()
                and matrix.abs().max() <= 1e6 and gain_weight.abs().max() <= 1e6)
            trials.append(row)
            key = q['validation_paired_HOST_causal_score_MSE']
            if diagnostic is None or key < diagnostic[0]:
                diagnostic = key, row
            if best_this_profile is None or key < best_this_profile['validation_paired_HOST_causal_score_MSE']:
                best_this_profile = row
            if row['eligible'] and (best is None or key < best[0]):
                best = key, actual_matrix.clone(), gain_weight.clone(), row
        print(json.dumps({'source_penalty': penalty, 'asymptotic_margin': asymptotic,
                          'best_profile_trial': best_this_profile}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'baseline': baseline, 'source_penalty_trials': source_penalties,
          'asymptotic_margin_trials': asymptotic_margins, 'energy_ridge_trials': energy_ridges,
          'fixed_source_covariance_ridge': 1e-6, 'source_positions': positions,
          'source_direction_training_windows': len(source_train), 'source_direction_holdback_windows': len(source_valid),
          'source_routing_bound_windows': len(SX), 'training_word_windows': 192, 'validation_word_windows': 64,
          'source_geometry': source_reports, 'trials': trials, 'eligible': best is not None,
          'selected': None if best is None else best[-1], 'best_diagnostic': None if diagnostic is None else diagnostic[-1],
          'targets': 'Exact source-position means; original same-input HOST raw query energies; native channel importance from original paired HOST inputs',
          'construction': 'All256 source channels; finite native-floor gain allocation; one all-row original-energy solve in the exact four-head programme-energy nullspace; final analytic routing amplitude only',
          'no_iterative_refit_gradients_native_activation_softmax_whole_answers_or_model': True,
          'scope': 'Exact pre-layer3 input provenance; local query/key algebra only. Full model scoring remains necessary.',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py'),
                      'experiments/derive_a286_dense_native_key_query_v1.py': sha(ROOT / 'experiments/derive_a286_dense_native_key_query_v1.py'),
                      'experiments/derive_a291_energy_constrained_dense_query_v1.py': sha(ROOT / 'experiments/derive_a291_energy_constrained_dense_query_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp,
                         control_path, original.parent / 'circuit.pt', base.with_suffix('.manifest.json')] + historical_paths},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a292v1_positive_energy_constrained_query.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'query_norm': best[2], 'report': report})
print(json.dumps({'eligible': best is not None, 'selected': report['selected'],
                  'best_diagnostic': report['best_diagnostic'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
