"""Stabilize A82's programme-only norm with a quiet native-gain raw anchor.

The anchor is written only by common programme features, so it adds no native
word energy carrier. Three emitter features and the eleven existing scratch
cancellations suffice. Constant anchor gains[1,2,4,8,16] are derived/tested on
literal digit geometry and recorded current registers, never sum answers.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a76v1_pool96/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
gp = ROOT / 'artifacts/a82_linear_digit_emitter_v1.pt'
geos = torch.load(gp, weights_only=True)
fp = ROOT / 'artifacts/a76v1_syntax_stress.pt'
f = torch.load(fp, weights_only=True, mmap=True)
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
gh = 1 + h['model.layers.17.post_feedforward_layernorm.weight'].double()
anchor = int(gh.abs().argmin())
assert anchor == 439
units = c['units'].long()
raw = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
F = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
factor = c['piece']['source_flag_scale'] / raw[:, c['piece']['source_flag_coordinate']]
digit = raw[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
nl = raw[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
expected = torch.tensor(c['piece']['token_ids'])[torch.where(nl > .5, torch.full_like(nl, 10, dtype=torch.long), digit.round().long().clamp(0, 9))]
common = F[:, units[0]] + F[:, units[1]]
cancel = F[:, units[20:]] @ ch['model.layers.17.mlp.down_proj.weight'][:, units[20:]].double().T
records, candidates = [], {}
for count in (96, 128):
    g = geos['candidates'][str(count)]
    pool = g['coordinates'].long()
    assert anchor not in pool.tolist()
    constant, linear, newline = [g[k].double() for k in ('constant', 'linear', 'newline')]
    minimum_point = float(-(constant @ linear) / linear.square().sum())
    minimum_point = max(-5., min(5., minimum_point))
    norm_candidates = [(constant + x * linear).square().sum() for x in (-5., 5., minimum_point)] + [newline.square().sum()]
    norm2_min = float(torch.stack(norm_candidates).min())
    norm2_max = float(torch.stack(norm_candidates).max())
    for amplitude in (1., 2., 4., 8., 16.):
        cancellation_scale = (amplitude ** 2 + .5 * (norm2_min + norm2_max)) ** .5
        programme = cancellation_scale * cancel
        programme[:, pool] += c['piece']['output_gain'] * (
            F[:, units[0], None] * constant[None]
            + common[:, None] * (digit - 4.5 * (1 - nl))[:, None] * linear[None]
            + F[:, units[1], None] * newline[None])
        programme[:, anchor] += c['piece']['output_gain'] * amplitude * common
        programme /= max(1., amplitude)
        gain = gh.clone()
        gain[pool] = 1e6
        out = raw + programme / (programme.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
        out /= (out.square().mean(1, keepdim=True) + 1e-6).sqrt()
        desired = (out * fg * E[expected].double()).sum(1)
        other = torch.full_like(desired, -torch.inf)
        for start in range(0, len(E), 4096):
            scores = out @ (E[start:start + 4096].double() * fg).T
            local = expected - start
            rows = ((local >= 0) & (local < scores.shape[1])).nonzero().flatten()
            scores[rows, local[rows]] = -torch.inf
            other = torch.maximum(other, scores.amax(1))
        margin = desired - other
        rec = {'pool_count': count, 'class_bias': g['report']['class_bias'],
               'anchor_amplitude': amplitude, 'anchor_coordinate': anchor,
               'anchor_native_post_gain': float(gh[anchor]),
               'cancellation_scale': cancellation_scale,
               'literal_direction_norm2_min': norm2_min, 'literal_direction_norm2_max': norm2_max,
               'raw_output_gain': c['piece']['output_gain'] / max(1., amplitude),
               'minimum_current_register_margin': float(margin.min()),
               'wrong_current_register_outputs': int((margin <= 0).sum())}
        records.append(rec)
        candidates[f'{count}_{amplitude:g}'] = {'coordinates': pool, 'constant': constant, 'linear': linear,
                                               'newline': newline, 'report': rec}
        print(json.dumps(rec), flush=True)
report = {'anchor_amplitude_trials': [1, 2, 4, 8, 16], 'pool_counts': [96, 128],
          'class_bias': 32, 'emitter_units': 3, 'cancellation_units': 11,
          'trials': records, 'anchor_changes_only_programme_columns': True,
          'limitation': 'Dominant programme component omits native MLP contribution; no checkpoint or actual execution yet.',
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, gp, fp)}, 'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a86_anchored_linear_emitter_v1.json', report)
atomic_torch(ROOT / 'artifacts/a86_anchored_linear_emitter_v1.pt', {'candidates': candidates, 'report': report})
