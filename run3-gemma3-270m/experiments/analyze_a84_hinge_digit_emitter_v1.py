"""Literal continuous digit interpolation with eleven emitter features.

Eight hinges at integers1..8 plus constant/digit/newline give unit-length
directions at each integer digit. This addresses A82's varying norm without
an answer fit. This static component uses ideal hinges, recorded registers,
and existing cancellation; actual GELU bounds and native contributions are
left to a subsequent compiler and official execution.
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
gp = ROOT / 'artifacts/a63_sparse_output_geometry_v1.pt'
geo = torch.load(gp, weights_only=True)
fp = ROOT / 'artifacts/a76v1_syntax_stress.pt'
f = torch.load(fp, weights_only=True, mmap=True)
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
units = c['units'].long()
raw = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
F = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
factor = c['piece']['source_flag_scale'] / raw[:, c['piece']['source_flag_coordinate']]
digit = raw[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
nl = raw[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
expected = torch.tensor(geo['token_ids'])[torch.where(nl > .5, torch.full_like(nl, 10, dtype=torch.long), digit.round().long().clamp(0, 9))]
common = F[:, units[0]] + F[:, units[1]]
cancel = F[:, units[20:]] @ ch['model.layers.17.mlp.down_proj.weight'][:, units[20:]].double().T
records, saved = [], {}
for count in (32, 48, 64, 96, 128):
    g = geo['candidates'][f'2_{count}']
    pool, directions = g['coordinates'].long(), g['directions'].double()
    if not g['report']['all_correct']:
        records.append({'pool_count': count, 'eligible_literal_geometry': False})
        continue
    D0 = directions[:, 0]
    D1 = directions[:, 1] - D0
    second = directions[:, 2:10] - 2 * directions[:, 1:9] + directions[:, :8]
    hinges = (digit[:, None] - torch.arange(1, 9, dtype=torch.float64)[None]).clamp_min(0)
    local = (1 - nl[:, None]) * D0[None] + digit[:, None] * D1[None]
    local += (1 - nl[:, None]) * (hinges @ second.T) + nl[:, None] * directions[:, 10]
    programme = cancel.clone()
    programme[:, pool] += c['piece']['output_gain'] * common[:, None] * local
    gain = 1 + h['model.layers.17.post_feedforward_layernorm.weight'].double()
    gain[pool] = 1e6
    out = raw + programme / (programme.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
    out /= (out.square().mean(1, keepdim=True) + 1e-6).sqrt()
    desired = (out * fg * E[expected].double()).sum(1)
    other = torch.full_like(desired, -torch.inf)
    for start in range(0, len(E), 4096):
        z = out @ (E[start:start + 4096].double() * fg).T
        local_id = expected - start
        rows = ((local_id >= 0) & (local_id < z.shape[1])).nonzero().flatten()
        z[rows, local_id[rows]] = -torch.inf
        other = torch.maximum(other, z.amax(1))
    margin = desired - other
    rec = {'pool_count': count, 'eligible_literal_geometry': True,
           'minimum_current_register_margin': float(margin.min()),
           'wrong_current_register_outputs': int((margin <= 0).sum()),
           'ideal_direction_norm_min': float(local.norm(dim=1).min()),
           'ideal_direction_norm_max': float(local.norm(dim=1).max())}
    records.append(rec)
    saved[str(count)] = {'coordinates': pool, 'directions': directions, 'constant': D0,
                         'linear': D1, 'second_differences': second, 'report': rec}
    print(json.dumps(rec), flush=True)
report = {'pool_count_trials': [32, 48, 64, 96, 128], 'hinge_knots': list(range(1, 9)),
          'emitter_units': 11, 'cancellation_units': 11, 'trials': records,
          'limitation': 'Ideal literal hinge component, omitting native contribution and GELU transition error.',
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, gp, fp)}, 'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a84_hinge_digit_emitter_v1.json', report)
atomic_torch(ROOT / 'artifacts/a84_hinge_digit_emitter_v1.pt', {'candidates': saved, 'report': report})
