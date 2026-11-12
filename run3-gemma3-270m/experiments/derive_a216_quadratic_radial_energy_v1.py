"""Quiet multi-row RMS reservoirs from original paired HOST final states.

A small quadratic feature solve approximates required raw anchor energy.
One PSD projection and rank truncation factor it into ordinary linear Down
rows. These are algebraic operations on recorded features, not new GLUs or
native nonlinear replay. Every target uses the fixed original parent.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified'] and cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base)
a, host, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, tp)]
assert torch.equal(a['inputs'], host['inputs']) and torch.equal(a['inputs'], top['inputs'][256:])
dn = 'model.layers.17.mlp.down_proj.weight'; pn = 'model.layers.17.post_feedforward_layernorm.weight'
units = torch.tensor(layout['native_restored_programme_units']['17'])
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
F = a['captures']['17.native_features'].reshape(-1, 2048).double(); F[:, units] = 0
W0 = ch[dn].double(); gain = 1 + ch[pn].double()
raw = F @ W0.T
before = a['captures']['17.raw'].reshape(-1, 640).double()
teacher = host['captures']['finalraw'].reshape(-1, 640).double()
def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
reference = rms(raw); teacher_rms = rms(teacher)
parent = before + raw / reference * gain
recorded = a['captures']['finalraw'].reshape(-1, 640).double()
reconstruction_error = float((parent - recorded).norm() / recorded.norm()); assert reconstruction_error < 1e-6
nt = 192 * 64
priority = (gain.abs() + 1e-6).square() / raw[:nt].square().mean(0).clamp_min(1e-15)
priority[gain.abs() > 1] = torch.inf
order = priority.argsort(); quiet = order[torch.isfinite(priority[order])]
declared_counts = [1, 2, 4, 8, 16]
counts = [n for n in declared_counts if n <= len(quiet)]
assert counts
print(json.dumps({'available_quiet_rows': len(quiet), 'available_anchor_counts': counts, 'skipped_unavailable_counts': [n for n in declared_counts if n not in counts]}), flush=True)
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
X = F[:, native] / reference
sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-15)
Z = X[:nt] / sd
covariance = Z.T @ Z / nt
ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2)
assert ev[-32] > 0
pca_map = vec[:, -32:] / sd[:, None] / ev[-32:].sqrt()[None]
del Z, covariance, ev, vec
total_energy = raw.square().sum(1)
numerator = raw * gain
desired_branch = teacher - before
total_cross = (numerator * desired_branch).sum(1)
total_numerator_energy = numerator.square().sum(1)
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
val_ids = torch.cat([top['topk_ids'][256 + 192:], a['topk_ids'][192:]], -1).reshape(-1, 16).long()
truth = top['topk_ids'][256 + 192:, :, 0].reshape(-1).long()
val_vectors = E[val_ids].double() * fg
Fv = F[nt:, native].clone(); del F
bm = json.loads(base.with_suffix('.manifest.json').read_text())
def quality(W, anchors):
    nr = raw[nt:].clone(); nr[:, anchors] = Fv @ W[anchors[:, None], native[None]].double().T
    nrms = rms(nr); state = before[nt:] + nr / nrms * gain
    scores = torch.einsum('nd,nkd->nk', state, val_vectors)
    winner = val_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
    return {'validation_restricted_matches': int((winner == truth).sum()),
        'validation_paired_HOST_state_relative_rms': float(((state - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_validation_raw_RMS_relative_change': float((nrms / reference[nt:] - 1).abs().max())}, state, winner
baseline, _, baseline_ids = quality(ch[dn], quiet[:0])
recorded_ids = a['topk_ids'][192:, :, 0].reshape(-1).long()
baseline['recorded_parent_matches'] = int((recorded_ids == truth).sum())
baseline['reconstructed_vs_recorded_prediction_changes'] = int((baseline_ids != recorded_ids).sum())
assert baseline['reconstructed_vs_recorded_prediction_changes'] <= 2
trials, target_diagnostics, best = [], [], None
for count in counts:
    anchors = quiet[:count]
    other_energy = (total_energy - raw[:, anchors].square().sum(1)).clamp_min(0)
    cross = total_cross - (numerator[:, anchors] * desired_branch[:, anchors]).sum(1)
    energy = total_numerator_energy - numerator[:, anchors].square().sum(1)
    assert energy.min() > 0
    unconstrained_alpha = cross / energy
    # Declare a finite radial domain relative to the fixed original parent.
    # The raw energy of all unchanged coordinates imposes a tighter cap.
    alpha_lower = .5 / reference[:, 0]
    alpha_upper = torch.minimum(2. / reference[:, 0], (640 / (other_energy + 640e-6)).sqrt())
    assert (alpha_upper >= alpha_lower).all()
    alpha = torch.maximum(alpha_lower, torch.minimum(unconstrained_alpha, alpha_upper))
    room = (640 / alpha.square() - other_energy - 640e-6).clamp_min(0)
    y = room[:nt] / reference[:nt, 0].square()
    target_diagnostics.append({'anchor_count': count, 'anchors': anchors.tolist(),
        'anchor_output_gains': gain[anchors].tolist(),
        'nonpositive_training_radial_optima': int((unconstrained_alpha[:nt] <= 0).sum()),
        'training_lower_bound_caps': int((unconstrained_alpha[:nt] < alpha_lower[:nt]).sum()),
        'training_upper_bound_caps': int((unconstrained_alpha[:nt] > alpha_upper[:nt]).sum()),
        'target_energy_mean': float(y.mean()),
        'target_scope': 'Positive radial projection of original paired HOST final residual onto fixed non-anchor parent numerator; quiet anchor output terms excluded, then actual full state checked'})
    # Include original anchor linear maps exactly alongside32 PCA directions.
    initial_map = torch.cat([pca_map, W0[anchors[:, None], native[None]].T], 1)
    mapped = X[:nt] @ initial_map
    gram = mapped.T @ mapped / nt
    be, bv = torch.linalg.eigh((gram + gram.T) / 2)
    keep = be > be[-1] * 1e-10
    basis = initial_map @ (bv[:, keep] / be[keep].sqrt()[None])
    B = X[:nt] @ basis
    first, second = torch.triu_indices(B.shape[1], B.shape[1])
    off_diagonal = first != second
    QX = B[:, first] * B[:, second]; QX[:, off_diagonal] *= 2 ** .5
    qsd = QX.square().mean(0).sqrt().clamp_min(1e-15); QZ = QX / qsd
    qcov = QZ.T @ QZ / nt; rhs = QZ.T @ y / nt
    qe, qv = torch.linalg.eigh((qcov + qcov.T) / 2); qe = qe.clamp_min(0)
    projected = qv.T @ rhs
    del mapped, gram, be, bv, QZ, qcov
    original_anchor_features = X[:nt] @ W0[anchors[:, None], native[None]].T
    for ridge in ridges:
        coefficients = (qv @ (projected / (qe + ridge))) / qsd
        quadratic = torch.zeros(B.shape[1], B.shape[1], dtype=torch.float64)
        quadratic[first, second] = coefficients
        quadratic[first[off_diagonal], second[off_diagonal]] /= 2 ** .5
        quadratic[second[off_diagonal], first[off_diagonal]] = quadratic[first[off_diagonal], second[off_diagonal]]
        pe, pv = torch.linalg.eigh(quadratic)
        chosen = torch.arange(len(pe) - min(count, len(pe)), len(pe))
        rows = (pv[:, chosen] * pe[chosen].clamp_min(0).sqrt()[None]).T @ basis.T
        if len(rows) < count: rows = torch.cat([rows, torch.zeros(count - len(rows), len(native))])
        new_features = X[:nt] @ rows.T
        left, _, right = torch.linalg.svd(original_anchor_features.T @ new_features, full_matrices=False)
        rows = (left @ right) @ rows
        fitted_energy = (X[:nt] @ rows.T).square().sum(1)
        energy_error = float((fitted_energy - y).square().mean().sqrt() / y.square().mean().sqrt())
        for strength in strengths:
            W = ch[dn].clone()
            W[anchors[:, None], native[None]] = (W0[anchors[:, None], native[None]]
                + strength * (rows - W0[anchors[:, None], native[None]])).float()
            assert torch.equal(W[:, units], ch[dn][:, units])
            changed = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            total = bm['changed_scalars'] - bm['per_tensor'][dn] + changed
            q, state, _ = quality(W, anchors)
            rec = {'anchor_count': count, 'anchors': anchors.tolist(), 'quadratic_ridge': ridge,
                'correction_strength': strength, 'quadratic_basis_dimension': B.shape[1],
                'quadratic_feature_count': len(coefficients), 'negative_quadratic_eigenvalues': int((pe < 0).sum()),
                'retained_positive_modes': int((pe[chosen] > 0).sum()), 'training_energy_relative_rms': energy_error,
                **q, 'maximum_down_weight': float(W.abs().max()), 'total_changed_scalars': total}
            rec['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6 and total <= 5016268
                and q['validation_restricted_matches'] >= baseline['recorded_parent_matches'] + 8
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
                and q['maximum_validation_raw_RMS_relative_change'] < .25)
            trials.append(rec)
            key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
            if rec['eligible'] and (best is None or key > best[0]): best = key, W.clone(), state.clone(), rec
    print(json.dumps({'anchor_count': count, 'selected': None if best is None else best[-1]}), flush=True)
    del QX, B, basis, qv, qe, quadratic, rows
full = None
if best is not None:
    state = (norm(best[2]) * fg).float()
    maximum = torch.full((len(state),), -torch.inf); winner = torch.full((len(state),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = state @ E[start:start + 4096].T
        values, indices = logits.max(1); improve = values > maximum
        maximum[improve] = values[improve]; winner[improve] = start + indices[improve]
    full = {'tokens': len(state), 'parent_recorded_matches': baseline['recorded_parent_matches'],
        'candidate_matches': int((winner == truth).sum()), 'required_extra_matches': 8}
    if full['candidate_matches'] < full['parent_recorded_matches'] + 8: best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'parent_final_relative_reconstruction_error': reconstruction_error,
    'training_windows': 192, 'validation_windows': 64, 'anchor_count_trials': declared_counts,
    'available_quiet_rows': len(quiet), 'executed_anchor_count_trials': counts,
    'unavailable_anchor_count_trials': [n for n in declared_counts if n not in counts],
    'quadratic_ridge_trials': ridges, 'correction_strength_trials': strengths,
    'fixed_PCA_directions': 32, 'additional_basis': 'Original linear anchor maps, whitened jointly with PCA',
    'relative_radial_inverse_RMS_domain': [.5, 2.], 'maximum_anchor_output_gain_magnitude': 1.,
    'quadratic_factorization': 'One spectral PSD projection, largest at-most-anchor-count positive modes, then orthogonal Procrustes alignment to original anchor outputs',
    'target_diagnostics': target_diagnostics, 'trials': trials, 'full_vocabulary_validation': full,
    'selected': None if best is None else best[-1], 'native_units': native.tolist(),
    'programme_columns_G_Up_and_all_gains_fixed': True,
    'format_caveat': 'No exact native-format features; full scoring required for any selected checkpoint',
    'no_answers_gradients_native_nonlinear_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, ap, lp, cp, hp, tp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a216v1_quadratic_radial_energy.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'down': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'full_vocabulary_validation': full,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
