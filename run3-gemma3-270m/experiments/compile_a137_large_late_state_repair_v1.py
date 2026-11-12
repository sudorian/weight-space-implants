"""A larger single original-HOST state repair in native layer16.

The A136 compact selector frees enough edits for up to1728 native columns.
Positive HOST-ray targets are derived once from paired unlabeled calibration
states and a fixed observed native normalizer. Each ridge/count fit starts
from HOST Down, not a prior fitted residual. Native features come solely from
the official meter. No gates, Up weights, gains or programme weights change.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, ELIGIBLE, sha

h = host_state()
base = ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'
wp = ROOT / 'artifacts/a137v1_late_basis.pt'
hp = ROOT / 'artifacts/a100v1_host16.pt'
lp = ROOT / 'artifacts/a137v1_late_basis_layout.json'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
w, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp)]
assert w['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'][:384], host['inputs'][:384])
assert all(not name.startswith('model.layers.16.') for name in ch)
for key, value in w['captures'].items():
    if key.startswith('word.') and key.endswith('program_features'):
        assert value.abs().max() == 0, key
dn = 'model.layers.16.mlp.down_proj.weight'
pn = 'model.layers.16.post_feedforward_layernorm.weight'
W0, gamma = h[dn].double(), 1 + h[pn].double()
assert gamma.abs().min() > 1e-9
F = w['captures']['word.16.features'].double().reshape(-1, 2048)
raw = w['captures']['word.16.down'].double().reshape(-1, 640)
before = w['captures']['word.16.raw'].double().reshape(-1, 640)
original = w['captures']['word.16.out'].double().reshape(-1, 640)
teacher = host['captures']['16.out'][:384].double().reshape(-1, 640)
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
raw_n = raw / rms
reference = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
linear_error = float((F @ W0.T - raw).norm() / raw.norm())
state_error = float((before + raw_n * gamma - original).norm() / original.norm())
assert linear_error < 1e-5 and state_error < 1e-5
nt = 256 * 64
training = torch.arange(len(F)) < nt
validation = ~training
def quality(r):
    z = r / (r.square().mean(1, keepdim=True) + 1e-6 / rms[validation].square()).sqrt()
    output = before[validation] + z * gamma
    error = (output - teacher[validation]) / reference[validation]
    direction = output / (output.square().mean(1, keepdim=True) + 1e-6).sqrt()
    direction -= teacher[validation] / reference[validation]
    return {'relative_hidden_state_rms': float(error.square().mean().sqrt()),
            'hidden_direction_rms': float(direction.square().mean().sqrt()),
            'relative_token_error_p99': float(error.square().mean(1).sqrt().quantile(.99))}
baseline = quality(raw_n[validation])
anchor_counts = [1, 4, 16, 64, 128]
donor_counts = [320, 640, 1024, 1536, 1728]
ridges = [.001, .01, .1, 1., 10.]
anchor_order = gamma.abs().argsort()
geometry = []
choice = None
for count in anchor_counts:
    anchors = anchor_order[:count]
    A, B = teacher / gamma, before / gamma
    A[:, anchors], B[:, anchors] = 0, 0
    radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
    aa = A.square().sum(1)
    centre = (A * B).sum(1) / aa
    minimum = (B - centre[:, None] * A).square().sum(1)
    half = ((radius - minimum).clamp_min(0) / aa).sqrt()
    lo = torch.maximum(centre - half, torch.full_like(centre, .5))
    hi = torch.minimum(centre + half, torch.full_like(centre, 2.))
    feasible = (minimum <= radius) & (lo <= hi)
    scale = torch.minimum(torch.maximum(torch.ones_like(lo), lo), hi)
    target = A * scale[:, None] - B
    room = (radius - target.square().sum(1)).clamp_min(0)
    direction = raw_n[:, anchors].clone()
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
    target[:, anchors] = direction * room.sqrt()[:, None]
    target[~feasible] = raw_n[~feasible]
    ideal = quality(target[validation])
    row = {'anchor_count': count, 'infeasible_rows': int((~feasible).sum()),
           'feasible_training_fraction': float(feasible[training].double().mean()),
           'ideal_geometry': ideal,
           'feasible_HOST_scale_quantiles': scale[feasible].quantile(
               torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist()}
    row['eligible'] = (row['feasible_training_fraction'] >= .95 and
                       ideal['hidden_direction_rms'] < .25 * baseline['hidden_direction_rms'])
    geometry.append(row)
    if row['eligible'] and choice is None:
        choice = target.clone(), feasible.clone(), row
report = {'layer': 16, 'baseline': baseline, 'anchor_count_trials': anchor_counts,
          'donor_count_trials': donor_counts, 'ridge_trials': ridges,
          'HOST_ray_scale_interval': [.5, 2.], 'anchor_geometry_trials': geometry,
          'linear_reconstruction_relative_error': linear_error,
          'state_reconstruction_relative_error': state_error,
          'geometry_selection': 'Smallest eligible anchor count, before any weight fit.',
          'minimum_direction_improvement_fraction': .05,
          'maximum_format_post_change': 100., 'trials': []}
out = ROOT / 'experiments/a137v1_large_late_state_repair/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'layer16_repair.json', report)
print(json.dumps({'baseline': baseline, 'anchor_geometry': geometry}), flush=True)
assert choice is not None, 'No eligible original HOST-ray target geometry'
target, feasible, selected_geometry = choice
used = training & feasible
Fn = F / rms / reference
standard = Fn[used].square().mean(0).sqrt().clamp_min(1e-12)
Z = Fn[used] / standard
Y = (target[used] - raw_n[used]) / reference[used]
cross = Z.T @ Y / len(Z)
priority = (cross * gamma).square().sum(1)
order = priority.argsort(descending=True)
columns_all = order[:max(donor_counts)]
Z = Z[:, columns_all]
covariance = Z.T @ Z / len(Z)
rhs = cross[columns_all]
del Z, Y, Fn, cross, target
PF = w['captures']['format.16.features'].double().reshape(-1, 2048)
PR = w['captures']['format.16.down'].double().reshape(-1, 640)
post0 = PR / (PR.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
best = None
for count in donor_counts:
    columns = order[:count]
    C = covariance[:count, :count]
    eigenvalues, V = torch.linalg.eigh((C + C.T) / 2)
    eigenvalues = eigenvalues.clamp_min(0)
    projected = V.T @ rhs[:count]
    for ridge in ridges:
        correction = (V @ (projected / (eigenvalues[:, None] + ridge))) / standard[columns, None]
        W = W0.clone()
        W[:, columns] += correction.T
        W = W.float()
        difference = W.double() - W0
        prediction = raw_n[validation] + (F[validation] / rms[validation]) @ difference.T
        metrics = quality(prediction)
        format_raw = PR + PF @ difference.T
        post = format_raw / (format_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
        format_change = float((post - post0).abs().max())
        changed = bm['changed_scalars'] + int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
        row = {'donor_count': count, 'ridge': ridge, **metrics,
               'maximum_format_post_change': format_change,
               'changed_scalars': changed, 'maximum_down_weight': float(W.abs().max())}
        row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6 and
                               changed <= int(.05 * ELIGIBLE) and format_change < 100 and
                               metrics['relative_token_error_p99'] < 1.1 * baseline['relative_token_error_p99'])
        report['trials'].append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and (best is None or metrics['hidden_direction_rms'] < best[0]):
            best = metrics['hidden_direction_rms'], W.clone(), row, columns.clone()
report.update(selected_geometry=selected_geometry, training_rows=int(used.sum()),
              selected=None if best is None else best[2],
              all_programme_weights_G_Up_and_gains_unchanged=True,
              no_answers_gradients_activation_replay_or_residual_refit=True)
atomic_json(out.parent / 'layer16_repair.json', report)
assert best is not None and best[0] < .95 * baseline['hidden_direction_rms'], 'Less than five percent local direction improvement'
ch[dn] = best[1]
c['large_late_state_repair_a137'] = report
atomic_torch(out.parent / 'circuit.pt', c)
categories = dict(bm['categories'])
categories[dn] = {'native_text_state_repair': int((best[1].view(torch.int32) != h[dn].view(torch.int32)).sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, lp],
     arguments={}, description=__doc__, extra=report, categories=categories)
