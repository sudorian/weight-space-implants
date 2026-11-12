"""Paired original-HOST state geometry at the exact current early MLP inputs.

Uses only factual recorded native features, fixed weights, linear maps and
RMS algebra. The original-HOST residual targets are derived from previously
recorded branch outputs and exactly reproduce the saved HOST final state.
No activation, attention, or unobserved model trajectory is replayed.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state(); base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
hp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
lp = ROOT / 'artifacts/a150v1_early_native_basis_layout.json'
ch = expand(h, base)
w, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp)]
layout = json.loads(lp.read_text())
assert w['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'][:256], host['inputs'])
assert host['report']['final_state_bit_equal']
nt = 192 * 64
report = {'training_windows': 192, 'validation_windows': 64, 'anchor_count_trials': [1, 4, 16, 64, 128],
          'HOST_ray_scale_interval': [.5, 2.], 'layers': {}, 'no_model_or_activation_replay': True}
for layer in range(1, 9):
    prefix = f'model.layers.{layer}.'
    dn, pn = prefix + 'mlp.down_proj.weight', prefix + 'post_feedforward_layernorm.weight'
    W = ch.get(dn, h[dn]).double(); gamma = 1 + ch.get(pn, h[pn]).double()
    units = torch.tensor(layout['programme_units'][str(layer)], dtype=torch.long)
    F = w['captures'][f'word.{layer}.features'].double().reshape(-1, 2048)
    assert F[:, units].abs().max() == 0
    before = w['captures'][f'word.{layer}.raw'].double().reshape(-1, 640)
    teacher = host['states'][f'{layer}.out'].double().reshape(-1, 640)
    raw = F @ W.T
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt(); raw_n = raw / rms
    ref = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    def quality(z):
        output = before[nt:] + z * gamma
        error = (output - teacher[nt:]) / ref[nt:]
        direction = output / (output.square().mean(1, keepdim=True) + 1e-6).sqrt() - teacher[nt:] / ref[nt:]
        return {'relative_hidden_state_rms': float(error.square().mean().sqrt()),
                'hidden_direction_rms': float(direction.square().mean().sqrt()),
                'relative_token_error_p99': float(error.square().mean(1).sqrt().quantile(.99))}
    baseline = quality(raw_n[nt:])
    zero = gamma.abs() < 1e-12
    divisor = torch.where(zero, torch.ones_like(gamma), gamma)
    order = gamma.abs().argsort()
    geometries = []
    for count in [1, 4, 16, 64, 128]:
        anchors = order[:max(count, int(zero.sum()))]
        A, B = teacher / divisor, before / divisor
        A[:, anchors], B[:, anchors] = 0, 0
        radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
        aa = A.square().sum(1).clamp_min(1e-30)
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
        length = direction.norm(dim=1, keepdim=True)
        empty = length[:, 0] < 1e-30
        direction[empty, 0] = 1
        direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
        target[:, anchors] = direction * room.sqrt()[:, None]
        target[~feasible] = raw_n[~feasible]
        ideal = quality(target[nt:])
        row = {'requested_anchor_count': count, 'actual_anchor_count': len(anchors),
               'anchors': anchors.tolist(), 'feasible_training_fraction': float(feasible[:nt].double().mean()),
               'infeasible_rows': int((~feasible).sum()), 'ideal': ideal,
               'feasible_scale_max_deviation': float((scale[feasible] - 1).abs().max()) if feasible.any() else None}
        row['eligible'] = (row['feasible_training_fraction'] >= .95 and
                           ideal['hidden_direction_rms'] < .25 * baseline['hidden_direction_rms'] and
                           ideal['relative_hidden_state_rms'] < .5 * baseline['relative_hidden_state_rms'])
        geometries.append(row)
    row = {'baseline': baseline, 'programme_units': units.tolist(), 'zero_gain_rows': zero.nonzero().flatten().tolist(),
           'geometry': geometries, 'eligible_anchor_counts': [g['requested_anchor_count'] for g in geometries if g['eligible']]}
    report['layers'][str(layer)] = row
    print(json.dumps({'layer': layer, 'baseline': baseline, 'geometry': geometries}), flush=True)
    atomic_json(ROOT / 'artifacts/a150v1_early_native_geometry.json', report)
report['inputs'] = {str(p.relative_to(ROOT)): sha(p) for p in (base, wp, hp, lp)}
report['sources'] = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                     'experiments/derive_a150_recorded_host_states_v1.py': sha(ROOT / 'experiments/derive_a150_recorded_host_states_v1.py')}
atomic_json(ROOT / 'artifacts/a150v1_early_native_geometry.json', report)
