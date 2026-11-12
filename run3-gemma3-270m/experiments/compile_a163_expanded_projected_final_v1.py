"""Project original HOST final-direction errors into a larger output subspace.

A136 frees the selector's excess native-repair parameters. This independent
construction uses its exact recorded final native features and fixed current
native map as a prior. Original paired HOST normalized final states supply
the only target. Unlike A138's fixed raw-state/anchor target, this uses the
HOST winner/runner-up metric and penalizes raw-numerator movement. No prior
failed fit is reused, no normalizer is updated during fitting, and no model,
native nonlinear activation, gradient or arithmetic-answer target is used.
"""
import copy
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha, ELIGIBLE

h = host_state()
base = ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'
wp = ROOT / 'artifacts/a137v1_late_basis.pt'
hp = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a83v1_host_top.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
w, host, top = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp, tp)]
assert w['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'][:384], host['inputs'][:384]) and torch.equal(w['inputs'][:384], top['inputs'][:384])
for key, value in w['captures'].items():
    if key.startswith('word.') and key.endswith('program_features'):
        assert value.abs().max() == 0, key
dn, pn = 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight'
W0 = ch[dn].double(); gamma = 1 + ch[pn].double()
units = c['units'].long(); strong = c['output_pool'].long()
assert len(strong) == 96
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
F = w['captures']['word.17.features'].double().reshape(-1, 2048)
raw = w['captures']['word.17.down'].double().reshape(-1, 640)
before = w['captures']['word.17.raw'].double().reshape(-1, 640)
current = w['captures']['word.17.out'].double().reshape(-1, 640)
teacher = host['captures']['finalraw'][:384].double().reshape(-1, 640)
def norm(x):
    return x / (x.square().mean(1, keepdim=True) + 1e-6).sqrt()
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
cr = (current.square().mean(1, keepdim=True) + 1e-6).sqrt()
assert (F @ W0.T - raw).norm() / raw.norm() < 1e-5
assert (before + norm(raw) * gamma - current).norm() / current.norm() < 1e-5
assert F[:, units].abs().max() == 0
teacher_n = norm(teacher); target = teacher_n - norm(current)
nt = 256 * 64
host_ids = top['topk_ids'][:384].reshape(-1, 8).long()
gap = (top['topk_values'][:384].reshape(-1, 8)[:, 0] - top['topk_values'][:384].reshape(-1, 8)[:, 1]).double().clamp_min(0)
weight = 1 / (gap + .1); weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
D = (E[host_ids[:, 0]].double() - E[host_ids[:, 1]].double()) * fg
metric = D[:nt].T @ (D[:nt] * weight[:nt, None]) / nt
metric = (metric + metric.T) / 2
metric_scale = float(metric.diagonal().mean())
Cstrong = metric[strong][:, strong].clone(); Cstrong.diagonal().add_(.001 * metric_scale)
conditional = metric - metric[:, strong] @ torch.linalg.solve(Cstrong, metric[strong])
priority = conditional.diagonal().clamp_min(0) * gamma.square()
priority[strong] = -torch.inf
priority[gamma.abs() < 1e-3] = -torch.inf
extra = priority.argsort(descending=True)
extra = extra[~torch.isin(extra, strong)]
order = torch.cat([strong, extra])
counts = [96, 256, 384, 512, 608]
ridges = [.001, .01, .1, 1., 10.]
metric_ridges = [0., .001, .01, .1, 1.]
strengths = [.25, .5, .75, 1., 1.25]
ordinary = torch.ones(640, dtype=torch.bool); ordinary[strong] = False
reference_gain2 = float(gamma[ordinary].square().median())
X = F[:nt, native] / rms[:nt] / cr[:nt] * weight[:nt, None].sqrt()
sd = X.square().mean(0).sqrt().clamp_min(1e-15)
Z = X / sd
C = Z.T @ Z / nt
rhs = Z.T @ (target[:nt] * weight[:nt, None].sqrt()) / nt
ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
projected = V.T @ rhs
del X, Z, C, rhs
truth = host_ids[nt:, 0]
rivals = E[host_ids[nt:]].double() * fg[None, None]
PF = w['captures']['format.17.features'][:, 8:13].double().reshape(-1, 2048)
PB = w['captures']['format.17.raw'][:, 8:13].double().reshape(-1, 640)
FR = PF @ W0.T
format_reference = norm(w['captures']['format.17.out'][:, 8:13].double().reshape(-1, 640))
assert w['logit_positions'] == [8, 9, 10, 11, 12]
format_margin = (w['topk_values'][384:, :, 0] - w['topk_values'][384:, :, 1]).double().reshape(-1)
maximum_readout_norm = 0.
for start in range(0, len(E), 4096):
    maximum_readout_norm = max(maximum_readout_norm, float((E[start:start + 4096].double() * fg).norm(dim=1).max()))
def quality(rows, delta):
    nr = raw[nt:].clone(); nr[:, rows] += F[nt:, native] @ delta.T
    nrms = (nr.square().mean(1, keepdim=True) + 1e-6).sqrt()
    state = before[nt:] + nr / nrms * gamma
    scores = torch.einsum('nd,nkd->nk', state, rivals)
    difference = norm(state) - teacher_n[nt:]
    boundary = (difference * D[nt:]).sum(1)
    fr = FR.clone(); fr[:, rows] += PF[:, native] @ delta.T
    final = norm(PB + norm(fr) * gamma)
    margin_bound = float((format_margin - 2 * (final - format_reference).norm(dim=1) * maximum_readout_norm).min())
    relative = (nrms / rms[nt:] - 1).abs().flatten()
    return {'validation_HOST_top8_matches': int((scores.argmax(1) == 0).sum()),
            'validation_boundary_score_rms': float((boundary.square() * weight[nt:]).sum().sqrt() / weight[nt:].sum().sqrt()),
            'validation_final_direction_rms': float(difference.square().mean().sqrt()),
            'maximum_relative_raw_norm_change': float(relative.max()),
            'p99_relative_raw_norm_change': float(relative.quantile(.99)),
            'minimum_recorded_format_margin_bound': margin_bound}, state
baseline, base_state = quality(strong, torch.zeros(len(strong), len(native), dtype=torch.float64))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
out = ROOT / 'experiments/a163v1_expanded_projected_final/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
trials, best = [], None
print(json.dumps({'baseline': baseline, 'base_changed_scalars': bm['changed_scalars']}), flush=True)
for count in counts:
    rows = order[:count]
    assert gamma[rows].abs().min() > 1e-3
    weight_penalty = 1 + reference_gain2 / gamma[rows].square()
    projections = {}
    for rho in metric_ridges:
        M = metric[rows][:, rows].clone(); M.diagonal().add_(rho * metric_scale * weight_penalty)
        projections[rho] = torch.linalg.solve(M, metric[rows]).T
    for ridge in ridges:
        beta = (V @ (projected / (ev[:, None] + ridge))) / sd[:, None]
        for rho in metric_ridges:
            correction = (beta @ projections[rho]).T / gamma[rows, None]
            for strength in strengths:
                fitted = (W0[rows[:, None], native[None]] + strength * correction).float()
                delta = fitted.double() - W0[rows[:, None], native[None]]
                q, state = quality(rows, delta)
                old_changes = int((ch[dn][rows[:, None], native[None]].view(torch.int32) != h[dn][rows[:, None], native[None]].view(torch.int32)).sum())
                new_changes = int((fitted.view(torch.int32) != h[dn][rows[:, None], native[None]].view(torch.int32)).sum())
                changed = bm['changed_scalars'] - old_changes + new_changes
                row = {'output_coordinate_count': count, 'native_ridge': ridge, 'boundary_metric_ridge_fraction': rho,
                       'correction_strength': strength, 'changed_scalars': changed, **q,
                       'maximum_native_down_weight': float(fitted.abs().max())}
                row['eligible'] = bool(changed <= int(.05 * ELIGIBLE) and fitted.isfinite().all() and fitted.abs().max() <= 1e6
                    and q['maximum_relative_raw_norm_change'] < .05 and q['p99_relative_raw_norm_change'] < .01
                    and q['minimum_recorded_format_margin_bound'] > 1
                    and q['validation_final_direction_rms'] < 1.2 * baseline['validation_final_direction_rms'])
                trials.append(row)
                key = (q['validation_HOST_top8_matches'], -q['validation_boundary_score_rms'])
                if row['eligible'] and (best is None or key > best[0]):
                    best = key, fitted.clone(), rows.clone(), state.clone(), row
            print(json.dumps({'output_coordinate_count': count, 'native_ridge': ridge, 'metric_ridge': rho,
                              'best': None if best is None else best[-1]}), flush=True)
        atomic_json(out.parent / 'partial_trials.json', {'baseline': baseline, 'trials': trials, 'selected': None if best is None else best[-1]})
report = {'baseline': baseline, 'output_coordinate_count_trials': counts, 'native_ridge_trials': ridges,
          'boundary_metric_ridge_fraction_trials': metric_ridges, 'correction_strength_trials': strengths,
          'training_windows': 256, 'validation_windows': 128, 'native_feature_count': len(native),
          'row_priority': 'HOST boundary-metric Schur diagonal times squared post gain, existing96writer coordinates first',
          'projection_penalty': 'rho*mean(metric_diagonal)*(1+median_ordinary_gain_squared/gain_squared)',
          'reference_ordinary_gain_squared': reference_gain2, 'p99_raw_norm_guard': .01, 'maximum_raw_norm_guard': .05,
          'target': 'Original paired HOST normalized final state; fixed A136map and original recorded raw/final RMS',
          'programme_G_Up_Down_and_all_gains_fixed': True, 'no_answers_gradients_or_iterative_residual_refits': True,
          'trials': trials, 'selected': None if best is None else best[-1]}
atomic_json(out.parent / 'expanded_projected_final.json', report)
assert best is not None, 'No candidate passes static bounds'
states = (norm(torch.cat([base_state, best[3]], 0)) * fg).float()
maxima = torch.full((len(states),), -torch.inf); winners = torch.full((len(states),), -1, dtype=torch.long)
for start in range(0, len(E), 4096):
    logits = states @ E[start:start + 4096].T
    values, ids = logits.max(1); better = values > maxima
    maxima[better] = values[better]; winners[better] = start + ids[better]
nv = len(truth)
old_matches = int((winners[:nv] == truth).sum()); new_matches = int((winners[nv:] == truth).sum())
report['full_vocabulary_validation'] = {'tokens': nv, 'baseline_matches': old_matches, 'candidate_matches': new_matches, 'required_extra_matches': 16}
report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out.parent / 'expanded_projected_final.json', report)
print(json.dumps({'full_vocabulary_validation': report['full_vocabulary_validation']}), flush=True)
assert new_matches >= old_matches + 16, 'Less than sixteen additional full-vocabulary matches'
_, fitted, rows, _, _ = best
W = ch[dn].clone(); W[rows[:, None], native[None]] = fitted
assert torch.equal(W[:, units], ch[dn][:, units])
ch[dn] = W; c['expanded_projected_final_a163'] = report
atomic_torch(out.parent / 'circuit.pt', c)
categories = copy.deepcopy(bm['categories'])
changed = W.view(torch.int32) != h[dn].view(torch.int32)
categories[dn] = {'arithmetic_programme': int(changed[:, units].sum()), 'native_text_repair': int(changed[:, native].sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / p for p in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), wp, hp, tp],
     arguments={}, description=__doc__, extra=report, categories=categories)
