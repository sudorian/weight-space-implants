"""One closed-form HOST boundary repair using at most1024 critical tokens on the larger A142 corpus.

The tensor-product kernel uses full native-feature covariance as its quiet
text metric. Its observations are original HOST/current mismatches and
near-boundary matches, all from unlabeled training calibration windows.
All targets, normalizers, rivals and features are fixed before any solve.
Only saved features, linear maps, RMS and frozen readout are evaluated.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
oldhp = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
calibration_base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ch = expand(h, base); recorded_model = expand(h, calibration_base)
circuit = torch.load(base.parent / 'circuit.pt', weights_only=True)
a, host, oldhost, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, oldhp, tp)]
assert a['checkpoint_sha256'] == sha(calibration_base)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256]) and torch.equal(a['inputs'][256:], host['inputs'])
for name in h:
    if name not in ['model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight']:
        assert torch.equal(ch.get(name, h[name]), recorded_model.get(name, h[name])), name
for key, value in a['captures'].items():
    if key.endswith('program_features'): assert value.abs().max() == 0, key
dn = 'model.layers.17.mlp.down_proj.weight'; pn = 'model.layers.17.post_feedforward_layernorm.weight'
units = circuit['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
pool = circuit['output_pool'].long(); assert len(pool) == 96 and len(native) == 2037
gain = 1 + ch[pn].double(); assert (gain[pool] == 1e6).all()
W0 = ch[dn].double()
F = a['captures']['17.features'].reshape(-1, 2048).double(); assert F[:, units].abs().max() == 0
before = a['captures']['17.raw'].reshape(-1, 640).double()
raw = F @ W0.T
def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
raw_rms = rms(raw)
parent = before + raw / raw_rms * gain
recorded = a['captures']['finalraw'].reshape(-1, 640).double()
recorded_raw = F @ recorded_model[dn].double().T
recorded_state = before + recorded_raw / rms(recorded_raw) * (1 + recorded_model[pn].double())
reconstruction_error = float((recorded_state - recorded).norm() / recorded.norm())
assert reconstruction_error < 1e-6
del recorded_model, recorded_raw, recorded_state
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
parent_rms, teacher_rms = rms(parent), rms(teacher)
target = teacher / teacher_rms - parent / parent_rms
host_ids = top['topk_ids'].reshape(-1, 8).long()
calibration_ids = a['topk_ids'].reshape(-1, 8).long()
all_ids = torch.cat([host_ids, calibration_ids], 1)
truth = host_ids[:, 0]
nt = 384 * 64
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
# Training rivals are explicitly restricted to original HOST/A123 top8.
# Their identity is fixed once, before any candidate; validation winners
# are checked against the full vocabulary only after a local candidate.
ranked = []
for start in range(0, len(parent), 256):
    candidates = all_ids[start:start + 256]
    vectors = E[candidates].double() * fg
    scores = torch.einsum('nd,nkd->nk', parent[start:start + 256], vectors)
    ranked.append(candidates.gather(1, scores.argsort(dim=1, descending=True)[:, :8]))
edited_ids = torch.cat(ranked)
del ranked, vectors, scores
rival = torch.where(truth != edited_ids[:, 0], edited_ids[:, 0], host_ids[:, 1])
D = (E[truth].double() - E[rival].double()) * fg
parent_margin = (parent / parent_rms * D).sum(1)
host_gap = (top['topk_values'].reshape(-1, 8)[:, 0] - top['topk_values'].reshape(-1, 8)[:, 1]).double()
wrong = (truth[:nt] != edited_ids[:nt, 0]).nonzero().flatten()
right = (truth[:nt] == edited_ids[:nt, 0]).nonzero().flatten()
wrong = wrong[parent_margin[wrong].argsort()[:512]]
right = right[host_gap[right].argsort()[:512]]
critical = torch.cat([wrong, right]); assert len(critical) <= 1024
X = F[:nt, native] / raw_rms[:nt] / parent_rms[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15)
Z = X / sd; covariance = Z.T @ Z / nt
ev, vectors = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
Zc = Z[critical].clone(); del X, Z, covariance
# Every output direction contributes to the original quiet-text metric,
# independently of which critical observations enter the boundary fit.
DR = (E[host_ids[:nt, 0]].double() - E[host_ids[:nt, 1]].double()) * fg
DR = DR[:, pool]
output_metric = DR.T @ DR / nt
output_scale = float(output_metric.diagonal().mean())
output_metric.diagonal().add_(.01 * output_scale)
Dc = D[critical][:, pool]
Do = torch.linalg.solve(output_metric, Dc.T).T
output_gram = Dc @ Do.T
y = (target[critical] * D[critical]).sum(1)
del target, DR, output_metric, D

input_ridges = [1e-6, 1e-4, .01, 1., 100.]
dual_ridges = [.001, .01, .1, 1., 10.]
strengths = [.25, .5, .75, 1., 1.25]
val_ids = all_ids[nt:]
val_truth = truth[nt:]
val_vectors = E[val_ids].double() * fg
Fv = F[nt:, native].clone(); del F
teacher_error = float(((parent[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt())
base_rows = W0[pool[:, None], native[None]]
bm = json.loads(base.with_suffix('.manifest.json').read_text())
def quality(rows):
    difference = rows.double() - base_rows
    nr = raw[nt:].clone(); nr[:, pool] += Fv @ difference.T
    new_rms = rms(nr); state = before[nt:] + nr / new_rms * gain
    scores = torch.einsum('nd,nkd->nk', state, val_vectors)
    winner = val_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
    return {'validation_restricted_matches': int((winner == val_truth).sum()),
        'validation_paired_HOST_state_relative_rms': float(((state - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_validation_raw_RMS_relative_change': float((new_rms / raw_rms[nt:] - 1).abs().max())}, state, winner
baseline, _, baseline_ids = quality(base_rows.float())
recorded_ids = edited_ids[nt:, 0]
baseline['restricted_parent_matches'] = int((recorded_ids == val_truth).sum())
baseline['restricted_readout_internal_prediction_changes'] = int((baseline_ids != recorded_ids).sum())
assert baseline['restricted_readout_internal_prediction_changes'] <= 2
trials, best = [], None
print(json.dumps({'critical_mismatches': len(wrong), 'critical_matches': len(right), 'baseline': baseline}), flush=True)
for input_ridge in input_ridges:
    inverse_times_features = ((Zc @ vectors) / (ev + input_ridge)) @ vectors.T
    kernel = (Zc @ inverse_times_features.T) * output_gram
    kernel = (kernel + kernel.T) / 2
    kernel_scale = float(kernel.diagonal().mean()); assert kernel_scale > 0
    for dual_ridge in dual_ridges:
        regularized = kernel.clone(); regularized.diagonal().add_(dual_ridge * kernel_scale)
        alpha = torch.linalg.solve(regularized, y)
        beta = inverse_times_features.T @ (alpha[:, None] * Do)
        correction = beta.T / sd[None] / gain[pool, None]
        for strength in strengths:
            rows = (base_rows + strength * correction).float()
            q, state, _ = quality(rows)
            W = ch[dn].clone(); W[pool[:, None], native[None]] = rows
            assert torch.equal(W[:, units], ch[dn][:, units])
            changed = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            total = bm['changed_scalars'] - bm['per_tensor'][dn] + changed
            rec = {'native_covariance_ridge': input_ridge, 'dual_ridge_fraction': dual_ridge,
                'correction_strength': strength, 'kernel_mean_diagonal': kernel_scale, **q,
                'maximum_down_weight': float(W.abs().max()), 'total_changed_scalars': total}
            rec['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6 and total <= 5016268
                and q['validation_restricted_matches'] >= baseline['restricted_parent_matches'] + 16
                and q['validation_paired_HOST_state_relative_rms'] <= 1.05 * teacher_error
                and q['maximum_validation_raw_RMS_relative_change'] < .001)
            trials.append(rec)
            key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
            if rec['eligible'] and (best is None or key > best[0]): best = key, W.clone(), state.clone(), rec
    print(json.dumps({'native_covariance_ridge': input_ridge, 'selected': None if best is None else best[-1]}), flush=True)
full = None
if best is not None:
    state = (norm(torch.cat([parent[nt:], best[2]])) * fg).float()
    maximum = torch.full((len(state),), -torch.inf); winner = torch.full((len(state),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = state @ E[start:start + 4096].T
        values, indices = logits.max(1); improve = values > maximum
        maximum[improve] = values[improve]; winner[improve] = start + indices[improve]
    nv = len(val_truth)
    full = {'tokens': nv, 'parent_matches': int((winner[:nv] == val_truth).sum()),
        'candidate_matches': int((winner[nv:] == val_truth).sum()), 'required_extra_matches': 16}
    if full['candidate_matches'] < full['parent_matches'] + 16: best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'reconstructed_capture_parent_relative_error': reconstruction_error,
    'calibration_parent': str(calibration_base.relative_to(ROOT)),
    'candidate_parent_differs_only_at_final_Down_and_post_gain': True,
    'training_rival_scope': 'Highest parent score among original HOST/A123 top8 union; not claimed full-vocabulary parent argmax',
    'training_windows': 384, 'validation_windows': 128,
    'critical_mismatch_limit': 512, 'critical_near_boundary_match_limit': 512,
    'critical_mismatch_count': len(wrong), 'critical_match_count': len(right),
    'critical_training_token_indices': critical.tolist(), 'maximum_dual_kernel_order': 1024,
    'native_covariance_ridge_trials': input_ridges, 'dual_ridge_fraction_trials': dual_ridges,
    'correction_strength_trials': strengths, 'fixed_output_covariance_ridge_fraction': .01,
    'target': 'Original HOST winner-minus-restricted-parent-rival margins at fixed recorded-parent normalizers; HOST runner used on already-matching training tokens',
    'native_quiet_metric': 'Full covariance of all24576training native feature vectors, not diagonal-only standardization',
    'all_targets_rivals_features_normalizers_fixed_before_independent_solves': True,
    'arithmetic_G_Up_Down_columns_and_all_gains_unchanged': True,
    'output_pool': pool.tolist(), 'native_units': native.tolist(),
    'trials': trials, 'full_vocabulary_validation': full, 'selected': None if best is None else best[-1],
    'format_caveat': 'No exact current-parent native format features; full arithmetic scoring required if selected',
    'no_answers_gradients_native_nonlinear_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, calibration_base, ap, hp, oldhp, tp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a217v1_large_corpus_critical_kernel.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'down': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'full_vocabulary_validation': full,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
