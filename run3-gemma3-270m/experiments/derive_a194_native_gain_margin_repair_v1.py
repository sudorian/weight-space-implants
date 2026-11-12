"""One-shot final native-gain repair against original HOST logit margins.

Only final post-MLP gains outside every arithmetic output row may change.
The recorded parent features make the final residual affine in these gains.
Targets are original HOST calibration winner/rival margins at fixed parent
normalizers. Each ridge/strength is an independent original-target solve;
no sum labels, gradients, native nonlinear replay or iterative fitting occur.
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
certp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
layout = json.loads(lp.read_text()); certificate = json.loads(certp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base)
a, host, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, tp)]
assert torch.equal(a['inputs'], host['inputs']) and torch.equal(a['inputs'], top['inputs'][256:])
dn = 'model.layers.17.mlp.down_proj.weight'; pn = 'model.layers.17.post_feedforward_layernorm.weight'
units = torch.tensor(layout['native_restored_programme_units']['17'])
fixed = ch[dn][:, units].ne(0).any(1); free = (~fixed).nonzero().flatten()
F = a['captures']['17.native_features'].reshape(-1, 2048).clone(); F[:, units] = 0
raw = F @ ch[dn].T; del F
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
z = fnorm(raw); del raw
before = a['captures']['17.raw'].reshape(-1, 640)
parameter0 = ch[pn]; gain0 = 1 + parameter0
parent = before + z * gain0
recorded_parent = a['captures']['finalraw'].reshape(-1, 640)
reconstruction_error = float((parent.double() - recorded_parent.double()).norm() / recorded_parent.double().norm())
assert reconstruction_error < 1e-5
teacher = host['captures']['finalraw'].reshape(-1, 640).double()
prms = (parent.double().square().mean(1, keepdim=True) + 1e-6).sqrt()
hrms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
delta_target = teacher / hrms - parent.double() / prms
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
ids = torch.cat([top['topk_ids'][256:], a['topk_ids']], -1).reshape(-1, 16).long()
truth = top['topk_ids'][256:, :, 0].reshape(-1).long()
nt = 192 * 64
gram = torch.zeros(len(free), len(free), dtype=torch.float64)
rhs = torch.zeros(len(free), dtype=torch.float64); observations = 0
for start in range(0, nt, 256):
    end = min(nt, start + 256); competitors = ids[start:end]
    directions = (E[truth[start:end]].double()[:, None] - E[competitors].double()) * fg
    valid = competitors != truth[start:end, None]
    for column in range(16):
        if column: valid[:, column] &= ~(competitors[:, :column] == competitors[:, column, None]).any(1)
    X = z[start:end, free].double()[:, None] * directions[:, :, free] / prms[start:end, None]
    y = (directions * delta_target[start:end, None]).sum(-1)
    X = X[valid]; y = y[valid]
    gram += X.T @ X; rhs += X.T @ y; observations += len(X)
gram /= observations; rhs /= observations
standard = gram.diagonal().sqrt().clamp_min(1e-12)
C = gram / (standard[:, None] * standard[None]); b = rhs / standard
ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0); projected_rhs = V.T @ b
del X, y, directions, delta_target, gram, rhs, C
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]
val_ids = ids[nt:]; val_truth = truth[nt:]
val_vectors = E[val_ids].double() * fg
parent_error = float(((parent[nt:].double() - teacher[nt:]) / hrms[nt:]).square().mean().sqrt())
def quality(parameter):
    state = before[nt:] + z[nt:] * (1 + parameter)
    scores = torch.einsum('nd,nkd->nk', state.double(), val_vectors)
    winner = val_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
    return {'validation_restricted_matches': int((winner == val_truth).sum()),
        'validation_paired_HOST_state_relative_rms': float(((state.double() - teacher[nt:]) / hrms[nt:]).square().mean().sqrt()),
        'validation_added_state_relative_rms': float(((state.double() - parent[nt:]) / hrms[nt:]).square().mean().sqrt())}, state, winner
baseline, _, baseline_ids = quality(parameter0)
recorded_ids = a['topk_ids'][192:, :, 0].reshape(-1).long()
baseline['recorded_parent_matches'] = int((recorded_ids == val_truth).sum())
baseline['reconstructed_vs_recorded_prediction_changes'] = int((baseline_ids != recorded_ids).sum())
assert baseline['reconstructed_vs_recorded_prediction_changes'] <= 2
trials, best = [], None
for ridge in ridges:
    correction = (V @ (projected_rhs / (ev + ridge))) / standard
    for strength in strengths:
        parameter = parameter0.clone(); parameter[free] = (parameter0[free].double() + strength * correction).float()
        assert torch.equal(parameter[fixed], parameter0[fixed])
        q, state, _ = quality(parameter)
        row = {'native_gain_ridge': ridge, 'correction_strength': strength, **q,
            'maximum_parameter': float(parameter.abs().max()), 'maximum_free_gain_change': float((parameter[free] - parameter0[free]).abs().max())}
        row['eligible'] = bool(parameter.isfinite().all() and parameter.abs().max() <= 1e6
            and q['validation_restricted_matches'] >= baseline['recorded_parent_matches'] + 8
            and q['validation_paired_HOST_state_relative_rms'] <= 1.05 * parent_error)
        trials.append(row)
        key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
        if row['eligible'] and (best is None or key > best[0]): best = key, parameter.clone(), state.clone(), row
    print(json.dumps({'ridge': ridge, 'best': None if best is None else best[-1]}), flush=True)
full = None
if best is not None:
    # Only the protected linear readout and RMS are applied to the exact
    # affine final-state candidate; no transformer component is replayed.
    x = fnorm(best[2]) * (1 + h['model.norm.weight'])
    maximum = torch.full((len(x),), -torch.inf); winner = torch.full((len(x),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = x @ E[start:start + 4096].T
        values, indices = logits.max(1); better = values > maximum
        maximum[better] = values[better]; winner[better] = start + indices[better]
    full = {'tokens': len(x), 'parent_recorded_matches': baseline['recorded_parent_matches'],
        'candidate_matches': int((winner == val_truth).sum()), 'required_extra_matches': 8}
    if full['candidate_matches'] < full['parent_recorded_matches'] + 8: best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'reconstructed_parent_relative_error': reconstruction_error, 'free_native_gain_rows': free.tolist(),
    'fixed_arithmetic_gain_rows': fixed.nonzero().flatten().tolist(), 'training_windows': 192, 'validation_windows': 64,
    'training_distinct_winner_rival_observations': observations, 'native_gain_ridge_trials': ridges,
    'correction_strength_trials': strengths, 'target': 'Original HOST winner/rival logit margins at fixed parent final-state RMS',
    'all_arithmetic_output_gains_fixed': True, 'trials': trials, 'full_vocabulary_validation': full,
    'selected': None if best is None else best[-1],
    'format_caveat': 'No exact current-parent final-format capture; an eligible candidate still requires official full arithmetic scoring',
    'no_answers_gradients_native_activation_replay_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ap, lp, certp, hp, tp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a194v1_native_gain_margin_repair.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'parameter': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'full_vocabulary_validation': full,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
