"""One original-HOST causal-score reconstruction after fixed pair deletion.

This programme-free diagnostic uses covariance and linear/RMS/RoPE algebra
on recorded inputs, with fixed original targets and no iterative refitting.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
control_path = ROOT / 'artifacts/a273v1_sparse_query_energy_accommodation.json'
ch, ref = expand(h, base), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3:
        TH = (norm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + norm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
state = w['captures']['word.2.raw'] + norm(e['captures']['word.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
TW = (norm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
del state, w, e, host
pre = 'model.layers.3.self_attn.'
qn, qgn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
QH = h[qn].double().reshape(4, 256, 640); KH = h[kn].double()
gh, kh = 1 + h[qgn].double(), 1 + h[kgn].double()
frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

def raw_query(t, Q):
    return torch.stack([t @ Q[i].T for i in range(4)])

host_qn, host_kn = norm(raw_query(TH, QH)), norm(TH @ KH.T)
qenergy = (host_qn[:, :192] * gh).square().mean((0, 1, 2))
kenergy = (host_kn[:192] * kh).square().mean((0, 1))
qe, ke = qenergy[:128] + qenergy[128:], kenergy[:128] + kenergy[128:]
qf = host_qn[:, :192].square().mean((0, 1, 2))
kf = host_kn[:192].square().mean((0, 1))
qf, kf = (qf[:128] + qf[128:]) / 256, (kf[:128] + kf[128:]) / 256
importance = qe * ke
importance += importance.mean() * (qf / qf.mean() + kf / kf.mean())
gaps = 1 - (frequency[:, None] * torch.arange(1, 9, dtype=torch.float64)).cos().amax(1)
cost = importance.clone(); cost[gaps < .02] = torch.inf
pairs = cost.argsort()[:5]
assert torch.isfinite(cost[pairs]).all()
host_scores = rotate(host_qn[:, 192:] * gh) @ rotate(host_kn[192:] * kh).transpose(-1, -2)[None] / 16
same_qn, same_kn = norm(raw_query(TW, QH)), norm(TW @ KH.T)
same_scores = rotate(same_qn[:, 192:] * gh) @ rotate(same_kn[192:] * kh).transpose(-1, -2)[None] / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / counts
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / counts).mean())

def quality(Q, K, gq, gk):
    qq = norm(raw_query(TW[192:], Q)) * gq
    kk = norm(TW[192:] @ K.T) * gk
    scores = rotate(qq) @ rotate(kk).transpose(-1, -2)[None] / 16
    return {'paired_HOST_causal_score_MSE': centered_mse(scores - host_scores),
            'same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores),
            'paired_HOST_query_RMS': float((qq - host_qn[:, 192:] * gh).square().mean().sqrt()),
            'paired_HOST_key_RMS': float((kk - host_kn[192:] * kh).square().mean().sqrt())}

baseline = quality(ch[qn].double().reshape(4, 256, 640), KH, 1 + ch[qgn].double(), kh)
control = json.loads(control_path.read_text())['baseline']
assert abs(baseline['paired_HOST_causal_score_MSE'] - control['validation_paired_HOST_causal_score_MSE']) < 1e-12
assert abs(baseline['same_input_HOST_causal_score_MSE'] - control['validation_same_input_HOST_causal_score_MSE']) < 1e-12
native = quality(QH, KH, gh, kh)
print(json.dumps({'baseline': baseline, 'unmodified_native_weights_on_current_inputs': native,
                  'selected_pairs': pairs.tolist()}), flush=True)
pair = int(pairs[0]); assert pair == 9
channels = torch.tensor([pair, pair + 128])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), channels)]
Q0, K = QH.clone(), KH.clone(); Q0[:, channels] = 0; K[channels] = 0
rotated_keys = rotate(norm(TW @ K.T) * kh)
same_keys = rotate(same_kn * kh)
nt = 192 * 64; T = TW.reshape(-1, 640)
input_count, output_count, basis_ridge = 64, 32, .001
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]

def inverse_rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs - torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

# Average of the original per-position causal-key covariance, before query
# rotation. Prefix outer moments are aggregated across windows, avoiding a
# windows x positions x 256 x 256 allocation.
global_cov = torch.zeros(256, 256, dtype=torch.float64)
prefix_outer = torch.zeros_like(global_cov)
prefix_sum = torch.zeros(192, 256, dtype=torch.float64)
for pos in range(64):
    row = rotated_keys[:192, pos]
    prefix_outer += row.T @ row / 192
    prefix_sum += row
    mean = prefix_sum / (pos + 1)
    rotated_cov = prefix_outer / (pos + 1) - mean.T @ mean / 192
    c, s = (pos * frequency).cos(), (pos * frequency).sin()
    R = torch.cat([torch.cat([c.diag(), -s.diag()], 1),
                   torch.cat([s.diag(), c.diag()], 1)], 0)
    global_cov += R.T @ rotated_cov @ R / (64 * 256)
global_ordinary_cov = global_cov[ordinary][:, ordinary]
global_ordinary_cov = (global_ordinary_cov + global_ordinary_cov.T) / 2
global_regularized = global_ordinary_cov.clone()
global_regularized.diagonal().add_(basis_ridge * global_ordinary_cov.diagonal().mean())
assert torch.linalg.eigvalsh(global_regularized).min() > 0
del prefix_outer, prefix_sum, mean, rotated_cov, R

selected_matrix = Q0.clone(); head_reports = []
for head in range(4):
    raw0 = T @ Q0[head].T
    reference_rms = (raw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    original_post = raw0 / reference_rms * gh
    original_scores = rotate(original_post.reshape(256, 64, 256)) @ rotated_keys.transpose(-1, -2) / 16
    original_native_scores = rotate(same_qn[head] * gh) @ same_keys.transpose(-1, -2) / 16
    targets = original_native_scores - original_scores
    centered_targets = (targets[:192] - (targets[:192] * causal).sum(-1)[..., None] / counts[:, None]) * causal
    cross_moment = inverse_rotate((centered_targets @ rotated_keys[:192]) / counts[:, None] / 16).reshape(nt, 256)
    # These are original score normal-equation moments, not a selected
    # candidate's residuals. One global solve defines a fixed basis only.
    moment_response = torch.zeros_like(cross_moment)
    moment_response[:, ordinary] = torch.linalg.solve(global_regularized, cross_moment[:, ordinary].T).T
    output_cov = moment_response[:, ordinary].T @ moment_response[:, ordinary] / nt
    oe, ov = torch.linalg.eigh((output_cov + output_cov.T) / 2)
    output_basis = torch.zeros(256, output_count, dtype=torch.float64)
    output_basis[ordinary] = ov[:, -output_count:].flip(1)
    X = T[:nt] / reference_rms[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
    C = Z.T @ Z / nt
    ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    inverse_root = (vec / (ev + basis_ridge).sqrt()[None]) @ vec.T
    cross = inverse_root @ (Z.T @ moment_response / nt)
    canonical, spectrum, _ = torch.linalg.svd(cross, full_matrices=False)
    input_map = inverse_root @ canonical[:, :input_count] / sd[:, None]
    features = X @ input_map
    rotated_basis = rotate(output_basis.T[:, None, :].expand(-1, 64, -1)).permute(1, 0, 2)
    key_features = torch.einsum('bsc,trc->btsr', rotated_keys[:192], rotated_basis) / 16
    key_features *= causal[None, :, :, None]
    mean_key = key_features.sum(2) / counts[None, :, None]
    H = torch.einsum('btsr,btsq->btrq', key_features, key_features) / counts[None, :, None, None]
    H -= mean_key[:, :, :, None] * mean_key[:, :, None, :]
    target_cross = torch.einsum('btsr,bts->btr', key_features, centered_targets) / counts[None, :, None]
    # Independent exact identity between the full original moments and their
    # output-basis projection guards inverse-RoPE orientation and centering.
    moment_check = float((target_cross.reshape(nt, output_count) - cross_moment @ output_basis).abs().max())
    covariance_check = float((H.mean((0, 1)) - output_basis.T @ global_cov @ output_basis).abs().max())
    assert moment_check < 1e-10 and covariance_check < 1e-10
    H = H.reshape(nt, output_count, output_count)
    gram4 = torch.empty(output_count, input_count, output_count, input_count, dtype=torch.float64)
    for r in range(output_count):
        for s in range(r, output_count):
            block = features.T @ (features * H[:, r, s, None]) / nt
            gram4[r, :, s, :] = block; gram4[s, :, r, :] = block.T
    rhs0 = target_cross.reshape(nt, output_count).T @ features / nt
    gram = gram4.reshape(output_count * input_count, output_count * input_count)
    scale = gram.diagonal().clamp_min(1e-30).sqrt()
    standardized = gram / scale[:, None] / scale[None]
    ce, cv = torch.linalg.eigh((standardized + standardized.T) / 2); ce = ce.clamp_min(0)
    rhs = cv.T @ (rhs0.reshape(-1) / scale)
    del key_features, H, target_cross, moment_response, X, Z, C, ev, vec, inverse_root, canonical, gram4, gram, standardized

    def head_quality(matrix):
        post = norm(TW[192:] @ matrix.T) * gh
        scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2) / 16
        return {'paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[head]),
                'same_input_HOST_causal_score_MSE': centered_mse(scores - original_native_scores[192:]),
                'paired_HOST_query_RMS': float((post - host_qn[head, 192:] * gh).square().mean().sqrt())}

    deletion_quality = head_quality(Q0[head])
    best = None; trials = []
    for ridge in ridges:
        beta = ((cv @ (rhs / (ce + ridge))) / scale).reshape(output_count, input_count)
        correction = torch.zeros_like(Q0[head])
        correction[ordinary] = (output_basis @ beta @ input_map.T)[ordinary] / gh[ordinary, None]
        assert torch.equal(correction[channels], torch.zeros_like(correction[channels]))
        for strength in strengths:
            candidate = (Q0[head] + strength * correction).float()
            quality_row = head_quality(candidate.double())
            eligible = bool(quality_row['paired_HOST_causal_score_MSE'] <= deletion_quality['paired_HOST_causal_score_MSE']
                            and quality_row['same_input_HOST_causal_score_MSE'] < deletion_quality['same_input_HOST_causal_score_MSE']
                            and candidate.isfinite().all() and candidate.abs().max() <= 1e6)
            rec = {'ridge': ridge, 'strength': strength, **quality_row, 'eligible': eligible,
                   'maximum_query_weight': float(candidate.abs().max())}
            trials.append(rec)
            if eligible and (best is None or rec['same_input_HOST_causal_score_MSE'] < best[0]):
                best = rec['same_input_HOST_causal_score_MSE'], candidate.clone(), rec
    if best is not None:
        selected_matrix[head] = best[1].double()
    row = {'head': head, 'deletion': deletion_quality, 'selected': None if best is None else best[2],
           'moment_identity_error': moment_check, 'covariance_identity_error': covariance_check,
           'output_moment_energy_fraction': float(oe[-output_count:].sum() / oe.sum()),
           'canonical_spectrum': spectrum.tolist(), 'trials': trials}
    head_reports.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('canonical_spectrum', 'trials')}), flush=True)
    del raw0, reference_rms, original_post, original_scores, original_native_scores, targets, centered_targets, cross_moment
    del output_cov, output_basis, features, input_map, cross, ce, cv, rhs, rhs0, scale

final_quality = quality(selected_matrix, K, gh, kh)
eligible = all(final_quality[key] <= .95 * baseline[key] for key in
               ('paired_HOST_causal_score_MSE', 'same_input_HOST_causal_score_MSE'))
report = {'analysis_only': True, 'pair': pair, 'baseline_A259': baseline,
          'deletion_control': quality(Q0, K, gh, kh), 'selected_quality': final_quality, 'eligible': eligible,
          'training_windows': 192, 'holdback_windows': 64, 'input_dimension': input_count, 'output_dimension': output_count,
          'fixed_basis_ridge': basis_ridge, 'metric_ridges': ridges, 'strengths': strengths,
          'head_reports': head_reports,
          'targets': 'Original same-input HOST causal scores; fixed pair-deletion normalization and key matrix',
          'scope': 'Programme-free native repair diagnostic; no complete reader or arithmetic/preservation claim',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/analyze_a283_single_pair_native_cost_v1.py': sha(ROOT / 'experiments/analyze_a283_single_pair_native_cost_v1.py'),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, wp, ep, hp, control_path,
                   ROOT / 'artifacts/a283v1_single_pair_native_cost.json']},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a284v1_single_pair_causal_repair.json'
atomic_json(out, report)
if eligible:
    from tools.common import atomic_torch
    atomic_torch(out.with_suffix('.pt'), {'query': selected_matrix.float(), 'key': K.float(), 'report': report})
print(json.dumps({'eligible': eligible, 'selected_quality': final_quality,
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
