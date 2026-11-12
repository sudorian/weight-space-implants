"""One original HOST state repair in the untouched native layer16 MLP.

Paired unlabeled calibration states define an exact positive HOST-ray target
on the existing post-MLP RMS sphere. Up to320native feature columns receive
one direct solve per declared ridge. Gates, Up, gains and all other weights
are fixed. Format bounds use only recorded native features, without GELU or
attention replay. There are no arithmetic answer targets or iterative fits.
"""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a98v1_pooled_gate_bounds/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'layer16_state_repair' not in c
bm = json.loads(base.with_suffix('.manifest.json').read_text())
wp, hp, fp = [ROOT / f'artifacts/a100v1_{name}.pt' for name in ('word16', 'host16', 'format16')]
w, host, fmt = [torch.load(path, weights_only=True, mmap=True) for path in (wp, hp, fp)]
assert torch.equal(w['inputs'], host['inputs'])
dn = 'model.layers.16.mlp.down_proj.weight'
pn = 'model.layers.16.post_feedforward_layernorm.weight'
assert all(not name.startswith('model.layers.16.') for name in ch), 'Layer16 must be untouched before this independent solve'
W0, gamma = h[dn].double(), 1 + h[pn].double()
assert gamma.abs().min() > 1e-9
F = w['captures']['16.features'].double().reshape(-1, 2048)
raw = w['captures']['16.down'].double().reshape(-1, 640)
before = w['captures']['16.raw'].double().reshape(-1, 640)
original = w['captures']['16.out'].double().reshape(-1, 640)
teacher = host['captures']['16.out'].double().reshape(-1, 640)
rms = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
raw_n = raw / rms
reference = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
linear_error = float((F @ W0.T - raw).square().mean().sqrt() / raw.square().mean().sqrt())
state_error = float((before + raw_n * gamma - original).square().mean().sqrt() / original.square().mean().sqrt())
assert linear_error < 1e-5 and state_error < 1e-5
train = torch.arange(len(F)) < 384 * 64
valid = ~train
Fn = F / rms / reference
standard = Fn[train].square().mean(0).sqrt().clamp_min(1e-12)
Z = Fn[train] / standard
original_error = (teacher - original) / reference
priority = (Z.T @ original_error[train] / len(Z)).square().sum(-1)
order = priority.argsort(descending=True)
del Z
anchor_order = gamma.abs().argsort()[:16]
PF = fmt['captures']['16.features'].double().reshape(-1, 2048)
PR = fmt['captures']['16.down'].double().reshape(-1, 640)
PB = fmt['captures']['16.raw'].double().reshape(-1, 640)
PS = fmt['captures']['16.out'].double().reshape(-1, 640)
post0 = PR / (PR.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
assert ((PB + post0 - PS).square().mean().sqrt() / PS.square().mean().sqrt()) < 1e-5
def quality(prediction_raw_normalized):
    normalized = prediction_raw_normalized / (prediction_raw_normalized.square().mean(-1, keepdim=True) + 1e-6 / rms[valid].square()).sqrt()
    output = before[valid] + normalized * gamma
    relative = (output - teacher[valid]) / reference[valid]
    direction = output / (output.square().mean(-1, keepdim=True) + 1e-6).sqrt() - teacher[valid] / reference[valid]
    return {'relative_hidden_state_rms': float(relative.square().mean().sqrt()),
            'hidden_direction_rms': float(direction.square().mean().sqrt()),
            'relative_token_error_p99': float(relative.square().mean(-1).sqrt().quantile(.99))}
baseline = quality(raw_n[valid])
print(json.dumps({'baseline': baseline, 'anchor_order': anchor_order.tolist()}), flush=True)
report = {'layer': 16, 'anchor_count_trials': [1, 2, 4, 8, 16],
          'donor_count_trials': [64, 128, 192, 256, 320],
          'ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'host_ray_scale_interval': [.5, 2.], 'maximum_format_post_change_bound': 100.,
          'minimum_local_direction_improvement_fraction': .05,
          'linear_reconstruction_relative_error': linear_error, 'state_reconstruction_relative_error': state_error,
          'baseline': baseline, 'anchor_order': anchor_order.tolist(), 'trials': []}
best = None
for anchor_count in (1, 2, 4, 8, 16):
    anchors = anchor_order[:anchor_count]
    A, B = teacher / gamma, before / gamma
    A[:, anchors], B[:, anchors] = 0, 0
    radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
    aa = A.square().sum(-1)
    centre = (A * B).sum(-1) / aa
    minimum = (B - centre[:, None] * A).square().sum(-1)
    halfwidth = ((radius - minimum).clamp_min(0) / aa).sqrt()
    low = torch.maximum(centre - halfwidth, torch.full_like(centre, .5))
    high = torch.minimum(centre + halfwidth, torch.full_like(centre, 2.))
    feasible = (minimum <= radius) & (low <= high)
    used = train & feasible
    scale = torch.minimum(torch.maximum(torch.ones_like(low), low), high)
    target = A * scale[:, None] - B
    room = (radius - target.square().sum(-1)).clamp_min(0)
    anchor_vector = raw_n[:, anchors].clone()
    anchor_vector /= anchor_vector.norm(dim=1, keepdim=True).clamp_min(1e-30)
    target[:, anchors] = anchor_vector * room.sqrt()[:, None]
    target[~feasible] = raw_n[~feasible]
    ideal = quality(target[valid])
    scale_quantiles = scale[feasible].quantile(torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist()
    Zu = Fn[used][:, order[:320]] / standard[order[:320]]
    Cu = Zu.T @ Zu / len(Zu)
    rhs = Zu.T @ ((target[used] - raw_n[used]) / reference[used]) / len(Zu)
    for count in (64, 128, 192, 256, 320):
        columns = order[:count]
        gram = Cu[:count, :count]
        ev, evec = torch.linalg.eigh((gram + gram.T) / 2)
        ev = ev.clamp_min(0)
        projected = evec.T @ rhs[:count]
        for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
            delta = (evec @ (projected / (ev[:, None] + ridge))) / standard[columns, None]
            W = W0.clone()
            W[:, columns] += delta.T
            Wf = W.float()
            difference = Wf.double() - W0
            predicted = raw_n[valid] + (F[valid] / rms[valid]) @ difference.T
            metrics = quality(predicted)
            format_raw = PR + PF @ difference.T
            format_post = format_raw / (format_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
            format_change = float((format_post - post0).abs().max())
            changed = bm['changed_scalars'] + int((Wf.view(torch.int32) != h[dn].view(torch.int32)).sum())
            row = {'anchor_count': anchor_count, 'donor_count': count, 'ridge': ridge,
                   'training_rows': int(used.sum()), 'infeasible_rows': int((~feasible).sum()),
                   'ideal_geometry': ideal, 'feasible_host_scale_quantiles': scale_quantiles,
                   **metrics, 'maximum_format_post_change': format_change,
                   'changed_scalars': changed, 'maximum_down_weight': float(Wf.abs().max())}
            row['eligible'] = bool(Wf.isfinite().all() and Wf.abs().max() <= 1e6
                                   and changed <= int(.05 * ELIGIBLE) and format_change < 100
                                   and int(used.sum()) > 24000)
            report['trials'].append(row)
            print(json.dumps(row), flush=True)
            if row['eligible'] and (best is None or row['hidden_direction_rms'] < best[0]):
                best = row['hidden_direction_rms'], Wf.clone(), row, columns.clone()
report['chosen'] = None if best is None else best[2]
report['source_sha256'] = sha(Path(__file__))
report['input_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in (base, wp, hp, fp)}
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'layer16_repair.json', report)
assert best is not None, 'No original HOST target fit within edit and format bounds'
assert best[0] < .95 * baseline['hidden_direction_rms'], 'Less than five percent local direction improvement; do not meter'
_, W, chosen, columns = best
ch[dn] = W
c['layer16_state_repair'] = report
atomic_torch(out.parent / 'circuit.pt', c)
categories = dict(bm['categories'])
categories[dn] = {'native_text_state_repair': int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
