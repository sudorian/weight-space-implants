"""One original-HOST causal query repair with fixed normalized-source keys.

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

control_path_new = ROOT / 'artifacts/a322v1_normalized_key_control.pt'
control = torch.load(control_path_new, weights_only=True)
control_report = control['report']
for path, digest in {**control_report['sources'], **control_report['inputs']}.items():
    assert sha(ROOT / path) == digest, path
assert control_report['original_A320_diagnostic_bit_exact']
positions = [3, 2, 1, 4]
Q0 = control['tensors'][qn].double().reshape(4, 256, 640)
gain_parameter = control['tensors'][gn]
gain = 1 + gain_parameter.double()
active = gain.abs() > 1e-8
K = control['tensors'][kn].double()
programme = control['programme_template'].double()
old_factors = control['original_programme_factors'].double()
native_control = control['native_without_programme'].double()
assert (native_control + old_factors[:, None, None] * programme - Q0).abs().max() < 1e-12
rotated_keys = rotate(norm(TW @ K.T) * kg)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
initial_quality = quality(Q0, gain)
original_diagnostic = control_report['best_diagnostic']
for key, value in initial_quality.items():
    assert abs(value - original_diagnostic[key]) < 1e-10, key
leader_quality = baseline
Pformat = current[:96, 8]
_, singular, vt = torch.linalg.svd(Pformat / Pformat.norm(dim=1, keepdim=True), full_matrices=True)
rank = int((singular > singular[0] * 1e-10).sum())
null = vt[rank:].T
assert rank == 96 and (Pformat @ null).abs().max() < 1e-8
input_counts = [8, 16, 32, 64, 128]
output_count = 16; basis_ridge = .001
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
head_reports = []; selected_matrix = Q0.clone(); any_selected = False
print(json.dumps({'A259_baseline': leader_quality, 'A320_reproduced': initial_quality, 'format_null_rank': rank}), flush=True)
for head in range(4):
    original_raw = T @ Q0[head].T
    reference_rms = rms(original_raw)
    original_post = original_raw / reference_rms * gain
    paired_post = host_queries[head].reshape(-1, 256)
    response = paired_post - original_post
    original_scores = rotate(original_post.reshape(256, 64, 256)) @ rotated_keys.transpose(-1, -2) / 16
    target_scores = host_scores[head] - original_scores
    X = (T[:nt] @ null) / reference_rms[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
    C = Z.T @ Z / nt
    ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    inverse_root = (vec / (ev + basis_ridge).sqrt()[None]) @ vec.T
    cross = inverse_root @ (Z.T @ response[:nt] / nt)
    canonical, spectrum, _ = torch.linalg.svd(cross, full_matrices=False)
    coordinate_map = inverse_root @ canonical[:, :max(input_counts)] / sd[:, None]
    input_map = null @ coordinate_map
    features = X @ coordinate_map
    ocov = response[:nt, active].T @ response[:nt, active] / nt
    oe, ov = torch.linalg.eigh((ocov + ocov.T) / 2)
    output_basis = torch.zeros(256, output_count, dtype=torch.float64)
    output_basis[active] = ov[:, -output_count:].flip(1)
    rotated_basis = rotate(output_basis.T[:, None, :].expand(-1, 64, -1)).permute(1, 0, 2)
    key_features = torch.einsum('bsc,trc->btsr', rotated_keys[:192], rotated_basis) / 16
    key_features *= causal[None, :, :, None]
    mean_key = key_features.sum(2) / count_causal[None, :, None]
    H = torch.einsum('btsr,btsq->btrq', key_features, key_features) / count_causal[None, :, None, None]
    H -= mean_key[:, :, :, None] * mean_key[:, :, None, :]
    target = target_scores[:192] * causal
    target_mean = target.sum(-1) / count_causal
    rhs_token = torch.einsum('btsr,bts->btr', key_features, target) / count_causal[None, :, None]
    rhs_token -= mean_key * target_mean[:, :, None]
    H = H.reshape(nt, output_count, output_count); rhs_token = rhs_token.reshape(nt, output_count)
    full_gram = torch.empty(output_count, max(input_counts), output_count, max(input_counts), dtype=torch.float64)
    for left in range(output_count):
        for right in range(left, output_count):
            block = features.T @ (features * H[:, left, right, None]) / nt
            full_gram[left, :, right, :] = block; full_gram[right, :, left, :] = block.T
    full_rhs = rhs_token.T @ features / nt
    del key_features, H, rhs_token, X, Z, C, ev, vec, cross, inverse_root, canonical
    others = torch.tensor([j for j in range(9) if j != positions[head]])
    difference = KR[:, positions[head], None] - KR[:, others]
    pr = SX[:, 8] @ programme[head].T
    Elinear = (difference * (pr * gain)[:, None]).sum(-1) / 16
    aa = pr.square().mean(1, keepdim=True)
    Aquad = Elinear.square() - 64 * aa
    assert Aquad.min() > 0 and Elinear.min() > 0

    def rebalance(correction):
        # One direct amplitude computation; neither target nor native fit is
        # updated after the arithmetic programme accommodation.
        native = native_control[head] + correction
        nr = SX[:, 8] @ native.T
        Dlinear = (difference * (nr * gain)[:, None]).sum(-1) / 16
        bb = 2 * (nr * pr).mean(1, keepdim=True)
        cc = nr.square().mean(1, keepdim=True) + 1e-6
        Bquad = 2 * Dlinear * Elinear - 64 * bb
        Cquad = Dlinear.square() - 64 * cc
        discriminant = Bquad.square() - 4 * Aquad * Cquad
        roots = torch.where(discriminant > 0,
            (-Bquad + discriminant.clamp_min(0).sqrt()) / (2 * Aquad), torch.zeros_like(Aquad))
        factor = float(torch.maximum(roots, -Dlinear / Elinear).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        matrix = (native + factor * programme[head]).float()
        return matrix, factor

    def head_quality(matrix):
        post = norm(TW[192:] @ matrix.T) * gain
        scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2) / 16
        sq = norm(SX[:, 8] @ matrix.T) * gain
        source_scores = (KR * sq[:, None]).sum(-1) / 16
        gaps = source_scores[:, positions[head]] - source_scores[:, others].amax(1)
        return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[head, 192:]),
            'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[head, 192:]),
            'validation_paired_HOST_query_RMS': float((post - host_queries[head, 192:]).square().mean().sqrt()),
            'minimum_source_routing_margin': float(gaps.min()),
            'current32_source_margin': float(gaps[96:128].min()), 'historical512_source_margin': float(gaps[-512:].min())}

    original_head = head_quality(Q0[head]); trials = []; best_head = None
    print(json.dumps({'head': head, 'baseline': original_head}), flush=True)
    for count in input_counts:
        gram = full_gram[:, :count, :, :count].reshape(output_count * count, output_count * count)
        scale = gram.diagonal().clamp_min(1e-30).sqrt()
        standardized = gram / scale[:, None] / scale[None]
        ce, cv = torch.linalg.eigh((standardized + standardized.T) / 2); ce = ce.clamp_min(0)
        rhs = cv.T @ (full_rhs[:, :count].reshape(-1) / scale)
        for ridge in ridges:
            beta = ((cv @ (rhs / (ce + ridge))) / scale).reshape(output_count, count)
            post_correction = output_basis @ beta @ input_map[:, :count].T
            correction = torch.zeros_like(Q0[head]); correction[active] = post_correction[active] / gain[active, None]
            null_error = float((Pformat @ correction.T).abs().max()); assert null_error < 1e-7
            for strength in strengths:
                matrix, factor = rebalance(strength * correction)
                q = head_quality(matrix.double())
                row = {'canonical_input_dimension': count, 'fixed_output_dimension': output_count,
                    'ridge': ridge, 'strength': strength, 'analytic_programme_factor': factor,
                    'original_programme_factor': float(old_factors[head]), 'real_native_format_null_error': null_error,
                    'maximum_query_weight': float(matrix.abs().max()), **q}
                row['eligible_head'] = bool(matrix.isfinite().all() and matrix.abs().max() <= 1e6
                    and q['minimum_source_routing_margin'] >= 7.999
                    and q['validation_paired_HOST_causal_score_MSE'] < original_head['validation_paired_HOST_causal_score_MSE']
                    and q['validation_same_input_HOST_causal_score_MSE'] <= original_head['validation_same_input_HOST_causal_score_MSE']
                    and q['validation_paired_HOST_query_RMS'] <= 1.05 * original_head['validation_paired_HOST_query_RMS'])
                trials.append(row)
                metric = q['validation_paired_HOST_causal_score_MSE']
                if row['eligible_head'] and (best_head is None or metric < best_head[0]):
                    best_head = metric, matrix.clone(), row
        print(json.dumps({'head': head, 'input_dimension': count, 'selected': None if best_head is None else best_head[-1]}), flush=True)
    if best_head is not None:
        selected_matrix[head] = best_head[1].double(); any_selected = True
    head_reports.append({'head': head, 'baseline': original_head, 'trials': trials,
        'selected': None if best_head is None else best_head[-1], 'original_response_spectrum': spectrum.tolist(),
        'original_response_energy_fraction_in_output_basis': float(oe[-output_count:].sum() / oe.sum().clamp_min(1e-30))})
    del full_gram, full_rhs, features, input_map, coordinate_map, ocov, output_basis
    del original_raw, reference_rms, original_post, paired_post, response, original_scores, target_scores
value = selected_matrix.reshape(1024, 640).float()
final_quality = quality(value.double().reshape(4, 256, 640), gain)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
tensors = {qn: value, gn: gain_parameter, kn: K.float()}
changed = bm['changed_scalars']
for name, tensor in tensors.items():
    changed += int((tensor.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'].get(name, 0)
maxvalue = max(float(t.abs().max()) for t in tensors.values())
eligible = bool(any_selected and changed <= 5016268 and maxvalue <= 1e6
    and final_quality['validation_paired_HOST_causal_score_MSE'] <= .9 * leader_quality['validation_paired_HOST_causal_score_MSE']
    and final_quality['validation_same_input_HOST_causal_score_MSE'] <= .9 * leader_quality['validation_same_input_HOST_causal_score_MSE']
    and final_quality['validation_paired_HOST_query_RMS'] <= 1.05 * leader_quality['validation_paired_HOST_query_RMS']
    and final_quality['minimum_source_routing_margin'] >= 7.999 and all(t.isfinite().all() for t in tensors.values()))
combined = {**final_quality, 'changed_scalars': changed, 'maximum_weight': maxvalue, 'eligible': eligible}
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'A259_baseline': leader_quality, 'A320_control': initial_quality, 'combined_candidate': combined,
    'selected': combined if eligible else None, 'head_reports': head_reports,
    'canonical_input_dimensions': input_counts, 'fixed_output_dimension': output_count,
    'whitening_ridge': basis_ridge, 'causal_score_ridges': ridges, 'strengths': strengths,
    'training_windows': 192, 'holdback_windows': 64, 'protected_current_format_rank': rank,
    'source_positions': positions, 'plus_decoder_not_yet_constructed': True,
    'construction': 'One original paired-HOST causal-score fit per candidate; fixed original normalizer/bases; direct programme amplitude accommodation afterward; no residual refit',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a237_causal_metric_query_repair_v1.py': sha(ROOT / 'experiments/derive_a237_causal_metric_query_repair_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp, control_path, control_path_new] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a322v1_fixed_key_causal_query_repair.json'
atomic_json(out, report)
if eligible: atomic_torch(out.with_suffix('.pt'), {'tensors': tensors, 'report': report})
print(json.dumps({'selected': report['selected'], 'combined': combined, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)

