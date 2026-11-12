"""Exact A343 diagnostic reconstructed for an explicit edit-efficiency review.

All computations are linear/RMS/QK/RoPE arithmetic on recorded inputs.
Native word covariance and source-format targets enter independent direct
solves. There is no native activation, softmax, model or gradient execution.
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
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
control_path = ROOT / 'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
historical_paths=[ROOT/f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
ch, ref = expand(h, base), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
del ref
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
historical=torch.cat([torch.load(p,weights_only=True,mmap=True)['captures']['3.input'][:,:9] for p in historical_paths]).double()
SX=torch.cat([current,historical]);nt=192*64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (kn, kgn): assert torch.equal(ch.get(name, h[name]), h[name])
QH = h[qn].double().reshape(4, 256, 640)
Qleader = ch[qn].double().reshape(4, 256, 640)
gh = 1 + h[gn].double(); g0 = 1 + ch[gn].double()
K = h[kn].double(); kg = 1 + h[kgn].double()
# A single unlabeled-word minimum-energy equals detector; never answer labels.
mean_equals=torch.cat([current[:96,8],historical[:1024,8]]).mean(0)
field_cov = T[:nt].T @ T[:nt] / nt
field_cov.diagonal().add_(1e-6 * field_cov.diagonal().mean())
qnull = torch.linalg.solve(field_cov, mean_equals)
qnull /= mean_equals @ qnull
precision = torch.linalg.inv(field_cov)
full_direction = precision @ mean_equals
support_order = (full_direction.square() / precision.diagonal().clamp_min(1e-30)).argsort(descending=True)
support_trials = [16,32,64,128,256]
native_candidates = []
for support_count in support_trials:
    columns = support_order[:support_count]
    direction = torch.zeros(640, dtype=torch.float64)
    direction[columns] = torch.linalg.solve(field_cov[columns][:, columns], mean_equals[columns])
    direction /= mean_equals @ direction
    native = QH - (mean_equals @ QH.transpose(-1, -2))[:, :, None] * direction
    null_error = float((mean_equals @ native.transpose(-1, -2)).abs().max())
    assert null_error < 1e-7
    outside = torch.arange(640)[~torch.isin(torch.arange(640), columns)]
    assert torch.equal(native[:, :, outside], QH[:, :, outside])
    native_candidates.append((native, outside, {'support_count':support_count,
        'support_columns':columns.tolist(), 'native_mean_null_error':null_error,
        'field_word_train_RMS':float((T[:nt] @ direction).square().mean().sqrt()),
        'field_word_holdback_RMS':float((T[nt:] @ direction).square().mean().sqrt()),
        'maximum_source_field_error':float((current[:,8]@direction-1).abs().max())}))
del precision
field_values = SX[:, 8] @ qnull
assert field_values.min() > 0
field_report = {'fixed_word_covariance_ridge': 1e-6,
    'training_equals_mean_field': float(mean_equals @ qnull),
    'minimum_current_equals_field': float(field_values[:128].min()),
    'minimum_historical_equals_field': float(field_values[128:].min()),
    'minimum_holdback32_equals_field': float(field_values[96:128].min()),
    'training_word_field_RMS': float((T[:nt] @ qnull).square().mean().sqrt()),
    'holdback_word_field_RMS': float((T[nt:] @ qnull).square().mean().sqrt())}
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
        'current32_source_margin': float(gaps[:, 96:128].min()), 'historical512_source_margin':float(gaps[:,-512:].min())}
baseline = quality(Qleader, g0)
control = json.loads(control_path.read_text())['combined_candidate']
assert abs(baseline['validation_paired_HOST_causal_score_MSE']-control['A237_paired_HOST_score_MSE'])<1e-12
positions=[3,2,1,4]
def source_amplitudes(native, program, gain, target_margin):
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
        A = E.square() - target_margin**2 * aa; Bq = 2 * D * E - target_margin**2 * bb; Cq = D.square() - target_margin**2 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Asymptotic source margin does not exceed target',
                          'minimum_asymptotic_source_margin': float((E / aa.clamp_min(1e-30).sqrt()).min())}
        discriminant = Bq.square() - 4 * A * Cq
        root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        gap = (D + factor * E) / (cc + factor * bb + factor**2 * aa).sqrt()
        assert gap.min() >= target_margin - .001
        factors.append(factor); gaps.append(float(gap.min()))
    return torch.tensor(factors, dtype=torch.float64), {'programme_factors': factors, 'analytic_source_margins': gaps}

source_train = torch.cat([torch.arange(96),torch.arange(128,128+1024)])
BOS = SX[source_train, 0]
_, singular, vt = torch.linalg.svd(BOS, full_matrices=True)
bos_rank = int((singular > singular[0] * 1e-10).sum())
null = vt[bos_rank:].T
assert null.shape[0] == 640 and null.shape[1] == 640 - bos_rank
word_features = ((T[:nt] / rms(T[:nt] @ K.T)) @ null).reshape(192, 64, -1)
original_rotated_queries = torch.stack([rotate(norm(TW[:192] @ QH[i].T) * gh) for i in range(4)])
word_gram_checks = []
pairs = [9, 15, 4, 17, 21]
format_masses = [.01, .1, 1., 10., 100.]
radii = [.5, .75, 1., 1.25, 1.5]
margin_trials = [12.,16.,24.,32.,48.]
headroom = 1.25
SXtrain = SX[source_train,1:9].reshape(-1,640)
raw_source = SXtrain @ K.T
trials = []; best = None; diagnostic = None
bm = json.loads(base.with_suffix('.manifest.json').read_text())
print(json.dumps({'baseline': baseline, 'BOS_null_rank': bos_rank, 'query_field': field_report}), flush=True)
def causal_key_gram(pair):
    # The coefficient order is [cos-row null coefficients, sin-row null coefficients].
    X = word_features
    dim = X.shape[-1]
    theta = torch.arange(64, dtype=torch.float64) * frequency[pair]
    co, si = theta.cos(), theta.sin()
    q0, q1 = original_rotated_queries[..., pair], original_rotated_queries[..., pair + 128]
    counts = count_causal[None, None]
    def backwards(x):
        return torch.flip(torch.cumsum(torch.flip(x, [-1]), -1), [-1])
    S00 = backwards(q0.square() / counts).sum(0) / (4 * 64 * 256)
    S11 = backwards(q1.square() / counts).sum(0) / (4 * 64 * 256)
    S01 = backwards(q0 * q1 / counts).sum(0) / (4 * 64 * 256)
    g0, g1 = kg[pair], kg[pair + 128]
    H00 = g0.square() * (co.square() * S00 + si.square() * S11 + 2 * co * si * S01)
    H11 = g1.square() * (si.square() * S00 + co.square() * S11 - 2 * co * si * S01)
    H01 = g0 * g1 * (co * si * (S11 - S00) + (co.square() - si.square()) * S01)
    flat = X.reshape(nt, dim)
    gram = torch.empty(2 * dim, 2 * dim, dtype=torch.float64)
    gram[:dim, :dim] = flat.T @ (flat * H00.reshape(-1, 1)) / 192
    gram[dim:, dim:] = flat.T @ (flat * H11.reshape(-1, 1)) / 192
    gram[:dim, dim:] = flat.T @ (flat * H01.reshape(-1, 1)) / 192
    gram[dim:, :dim] = gram[:dim, dim:].T
    XC = torch.cumsum(X * co[None, :, None], 1) / count_causal[None, :, None]
    XS = torch.cumsum(X * si[None, :, None], 1) / count_causal[None, :, None]
    for head in range(4):
        M0 = (q0[head, :, :, None] * XC + q1[head, :, :, None] * XS) * g0 / 16
        M1 = (-q0[head, :, :, None] * XS + q1[head, :, :, None] * XC) * g1 / 16
        M = torch.cat([M0, M1], -1).reshape(nt, 2 * dim)
        gram -= M.T @ M / (4 * nt)
    gram = (gram + gram.T) / 2
    probe = torch.sin(torch.arange(2 * dim, dtype=torch.float64) * 1.61803398875) * .001
    pair_delta = X @ probe.reshape(2, dim).T
    delta_key = torch.zeros(192, 64, 256, dtype=torch.float64)
    delta_key[..., pair] = pair_delta[..., 0] * g0
    delta_key[..., pair + 128] = pair_delta[..., 1] * g1
    scores = original_rotated_queries @ rotate(delta_key).transpose(-1, -2)[None] / 16
    direct = centered_mse(scores)
    quadratic = float(probe @ gram @ probe)
    relative = abs(quadratic - direct) / max(direct, 1e-30)
    assert relative < 1e-9, (pair, direct, quadratic, relative)
    check = {'pair': pair, 'direct_linear_causal_score_MSE': direct,
             'quadratic_form': quadratic, 'relative_error': relative,
             'mean_diagonal': float(gram.diagonal().mean())}
    word_gram_checks.append(check)
    print(json.dumps({'word_gram_check': check}), flush=True)
    return gram

for pair in pairs:
    channels = torch.tensor([pair, pair + 128])
    remaining = torch.arange(256)[~torch.isin(torch.arange(256), channels)]
    native_vector = norm(raw_source)[:, channels].mean(0)
    native_radius = float(native_vector.norm())
    assert native_radius > 1e-8
    post_gain_phase = torch.atan2(native_vector[1] * kg[channels[1]], native_vector[0] * kg[channels[0]])
    phase = (torch.tensor(positions, dtype=torch.float64) - 8) * frequency[pair] + post_gain_phase
    programme = torch.zeros_like(QH)
    programme[:, pair] = phase.cos()[:, None] * qnull
    programme[:, pair + 128] = phase.sin()[:, None] * qnull
    residual_rms = (raw_source[:, remaining].square().sum(-1, keepdim=True) / 256 + 1e-6).sqrt()
    raw_features = SXtrain / residual_rms
    XF = raw_features @ null
    format_cov = XF.T @ XF / len(XF)
    word_gram = causal_key_gram(pair)
    format_gram = torch.block_diag(format_cov, format_cov)
    trace_scale = word_gram.diagonal().mean() / format_gram.diagonal().mean()
    factors_by_mass = {}
    for mass in format_masses:
        covariance = word_gram + mass * trace_scale * format_gram
        covariance.diagonal().add_(1e-6 * word_gram.diagonal().mean())
        factors_by_mass[mass] = torch.linalg.cholesky(covariance)
    del covariance, format_gram, word_gram
    for radius in radii:
        v = radius * native_vector
        assert v.square().sum() < 256
        wanted = v / (1 - v.square().sum() / 256).sqrt()
        target = wanted[None] - raw_features @ K[channels].T
        format_rhs = XF.T @ target / len(XF)
        for mass in format_masses:
            rhs = mass * trace_scale * format_rhs.T.reshape(-1, 1)
            coefficients = torch.cholesky_solve(rhs, factors_by_mass[mass])
            delta = null @ coefficients.reshape(2, -1).T
            assert (BOS @ delta).abs().max() < 1e-8
            key = K.clone(); key[channels] += delta.T; key = key.float().double()
            bos_error = float(((SX[:, 0] @ (key - K).T).norm(dim=-1, keepdim=True) / rms(SX[:, 0] @ K.T)).max())
            raw_normalized = norm(SX @ key.T)
            SK = raw_normalized * kg
            KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
            rotated_keys = rotate(norm(TW @ key.T) * kg)
            key_rms = float((rotated_keys[192:] - host_keys[192:]).square().mean().sqrt())
            pair_error = raw_normalized[:, 1:9, channels] - v
            unit_gaps = []
            for head, pos in enumerate(positions):
                direction = norm(SX[:, 8] @ programme[head].T)
                scores = (KR * direction[:, None]).sum(-1) / 16
                others = torch.tensor([j for j in range(9) if j != pos])
                unit_gaps.append(scores[:, pos] - scores[:, others].amax(1))
            minimum_unit_gap = float(torch.stack(unit_gaps).min())
            for target_margin in margin_trials:
                row = {'pair': pair, 'native_radius_multiplier': radius, 'normalized_key_radius': float(v.norm()), 'original_native_pair_mean':native_vector.tolist(), 'original_native_pair_radius':native_radius, 'post_Kgain_native_phase':float(post_gain_phase), 'format_covariance_mass': mass,
                    'target_source_margin': target_margin, 'minimum_unit_gain_source_gap': minimum_unit_gap,
                    'rounded_BOS_raw_error_over_original_RMS': bos_error,
                    'source_normalized_pair_RMS_error': float(pair_error.square().mean().sqrt()),
                    'source_maximum_pair_error_over_target_radius': float(pair_error.norm(dim=-1).max() / v.norm()),
                    'validation_paired_HOST_key_RMS': key_rms}
                if minimum_unit_gap <= 0:
                    trials.append({**row, 'eligible': False, 'reason': 'Nonpositive asymptotic source geometry'}); continue
                query_gain = max(float(gh[channels].max()), headroom * target_margin / minimum_unit_gap)
                if query_gain > 1e6:
                    trials.append({**row, 'eligible': False, 'reason': 'Required pair gain exceeds cap'}); continue
                gain = gh.clone(); gain[channels] = query_gain
                gain_weight = (gain - 1).float(); gain = 1 + gain_weight.double()
                scale = torch.ones_like(gh); scale[channels] = gh[channels] / gain[channels]
                geometry_row = row
                for Bnative, other_columns, support_report in native_candidates:
                    row = dict(geometry_row)
                    native = Bnative * scale[None, :, None]
                    factors, routing = source_amplitudes(native, programme, gain, target_margin)
                    row.update(analytic_pair_query_gain=query_gain, routing=routing, **support_report)
                    if factors is None:
                        trials.append({**row, 'eligible': False}); continue
                    query = (native + factors[:, None, None] * programme).float()
                    assert torch.equal(query[:, remaining][:, :, other_columns], h[qn].reshape(4, 256, 640)[:, remaining][:, :, other_columns])
                    actual = quality(query.double(), gain)
                    tensors = {qn: query.reshape(1024, 640), gn: gain_weight, kn: key.float()}
                    total = bm['changed_scalars']
                    for name, tensor in tensors.items():
                        total += int((tensor.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'].get(name, 0)
                    maximum = max(float(t.abs().max()) for t in tensors.values())
                    row.update(actual, changed_scalars=total, maximum_query_key_or_gain_weight=maximum,
                               all254_nonprogramme_Qrows_HOST_bit_exact_outside_support=True)
                    row['eligible'] = bool(actual['validation_paired_HOST_causal_score_MSE'] <= baseline['validation_paired_HOST_causal_score_MSE']
                        and actual['validation_same_input_HOST_causal_score_MSE'] <= baseline['validation_same_input_HOST_causal_score_MSE']
                        and actual['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS']
                        and actual['minimum_source_routing_margin'] >= target_margin - .001 and total <= bm['changed_scalars'] - 300000
                        and bos_error <= 1e-5 and all(t.isfinite().all() for t in tensors.values()) and maximum <= 1e6)
                    trials.append(row)
                    metric = actual['validation_paired_HOST_causal_score_MSE']
                    if diagnostic is None or metric < diagnostic[0]: diagnostic = metric, row
                    if row['eligible'] and (best is None or metric < best[0]): best = metric, tensors, row
            print(json.dumps({'pair': pair, 'radius': radius, 'format_mass': mass,
                              'best_diagnostic': None if diagnostic is None else diagnostic[-1]}), flush=True)
original_path=ROOT/'artifacts/a343v1_sparse_native_phase_A_reader.json'
original=json.loads(original_path.read_text())
assert best is not None
reproduced=dict(best[-1]); reproduced['eligible']=False
assert reproduced==original['best_diagnostic']
report = {'original_A343_diagnostic_bit_exact':True, 'original_ten_percent_screen_rejection_retained':True, 'new_guard':'At least300000fewer edits with paired/same causalMSE no worse than A259; source/caps unchanged', 'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'BOS_null_rank': bos_rank, 'full_BOS_right_nullspace_dimension': null.shape[1], 'query_field': field_report,
    'word_gram_checks': word_gram_checks, 'pair_trials': pairs, 'native_radius_multiplier_trials': radii,
    'format_covariance_mass_trials': format_masses, 'target_margin_trials': margin_trials,
    'fixed_asymptotic_headroom': headroom, 'fixed_covariance_ridge': 1e-6,
    'source_positions': positions, 'training_word_windows': 192, 'holdback_word_windows': 64,
    'training_format_windows':1120,'current_holdback_format_windows':32,'historical_holdback_format_windows':512,'independent_historical_source_windows':len(historical),
    'source_scope_limit': 'Earlier historical pre3 population has unchanged B interface; altered downstream trajectory needs composition/full score',
    'native_Q_policy': 'Original HOST with independently covariance-selected sparse mean null; only two rows per head receive gain compensation and programme',
    'support_count_trials':support_trials, 'support_diagnostics':[n[2] for n in native_candidates],
    'trials': trials, 'selected': None if best is None else best[-1],
    'best_diagnostic': None if diagnostic is None else diagnostic[-1],
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/derive_a343_sparse_native_phase_A_reader_v1.py':sha(ROOT/'experiments/derive_a343_sparse_native_phase_A_reader_v1.py'),
        'experiments/derive_a337_native_phase_B_reader_v1.py':sha(ROOT/'experiments/derive_a337_native_phase_B_reader_v1.py'),
        'experiments/derive_a335_covariance_sparse_B_reader_v1.py': sha(ROOT / 'experiments/derive_a335_covariance_sparse_B_reader_v1.py'),
        'experiments/derive_a334_marker_column_B_reader_v1.py': sha(ROOT / 'experiments/derive_a334_marker_column_B_reader_v1.py'),
        'experiments/derive_a332_sparse_native_B_reader_v1.py': sha(ROOT / 'experiments/derive_a332_sparse_native_B_reader_v1.py'),
        'experiments/derive_a328_realized_key_margin_allocation_v1.py': sha(ROOT / 'experiments/derive_a328_realized_key_margin_allocation_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base,reference,wp,ep,hp,control_path,original_path]+historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a352v1_edit_efficient_A_reader.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'tensors': best[1], 'query_field': qnull, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
