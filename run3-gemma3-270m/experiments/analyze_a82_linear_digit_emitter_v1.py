"""Test a three-feature literal decimal/newline emitter on recorded registers.

Nearest-digit logits are affine in the scalar digit: B - j*j + 2*j*d.
No whole-problem examples, answers, model replay, gradients, or fitting are
used. Three common-gated linear features replace twenty threshold features.
The existing scratch cancellation is included. Native MLP contribution is
omitted in this component check and must be restored and officially checked.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a76v1_pool96/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
geometry_path = ROOT / 'artifacts/a63_sparse_output_geometry_v1.pt'
geometry = torch.load(geometry_path, weights_only=True)
readout_path = ROOT / 'artifacts/readout_v1.pt'
readout = torch.load(readout_path, weights_only=True)
capture_path = ROOT / 'artifacts/a76v1_syntax_stress.pt'
capture = torch.load(capture_path, weights_only=True, mmap=True)
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
ids = torch.tensor(geometry['token_ids'])
C = readout['vocabulary_metric'].double()
d = torch.arange(10, dtype=torch.float64) - 4.5
probes = torch.tensor([j + error for j in range(10) for error in (-.2, 0., .2)])
expected_probe = torch.cat([ids[:10].repeat_interleave(3), ids[10:]])

def margins(states, expected):
    normalized = states / (states.square().mean(1, keepdim=True) + 1e-6).sqrt()
    desired = (normalized * fg * E[expected].double()).sum(1)
    other = torch.full_like(desired, -torch.inf)
    for start in range(0, len(E), 4096):
        z = normalized @ (E[start:start + 4096].double() * fg).T
        local = expected - start
        rows = ((local >= 0) & (local < z.shape[1])).nonzero().flatten()
        z[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, z.amax(1))
    return desired - other

raw = capture['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
features = capture['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
program = c['units'].long()
factor = c['piece']['source_flag_scale'] / raw[:, c['piece']['source_flag_coordinate']]
digit = raw[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = raw[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
label = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
expected = ids[label]
common = features[:, program[0]] + features[:, program[1]]
assert common.min() > 0
cancel = features[:, program[20:]] @ ch['model.layers.17.mlp.down_proj.weight'][:, program[20:]].double().T

records = []
saved = {}
for count in (32, 48, 64, 96, 128):
    pool = geometry['candidates'][f'2_{count}']['coordinates'].long()
    D = E[ids][:, pool].double() * fg[pool]
    B = torch.linalg.solve(C[pool][:, pool], D.T)
    W = torch.linalg.solve(D @ B, B.T).T
    frames, params = [], []
    for bias in (16., 32., 64., 128., 256.):
        constant = W[:, :10] @ (bias - d.square())
        linear = W[:, :10] @ (2 * d)
        scale = max(float((constant + linear * x).norm()) for x in (-4.7, 4.7))
        constant, linear = constant / scale, linear / scale
        nl = W[:, 10] / W[:, 10].norm()
        literal = torch.cat([constant[None] + (probes.double() - 4.5)[:, None] * linear[None], nl[None]])
        full = torch.zeros(len(literal), 640, dtype=torch.float64)
        full[:, pool] = literal
        frames.append(full)
        params.append((bias, constant, linear, nl))
    literal_margins = margins(torch.cat(frames), expected_probe.repeat(5)).reshape(5, -1)
    rows = []
    for i, (bias, constant, linear, nl) in enumerate(params):
        row = {
            'coordinate_count': count, 'class_bias': bias,
            'literal_minimum_full_vocabulary_margin': float(literal_margins[i].min()),
            'literal_integer_minimum_margin': float(literal_margins[i, :30][1::3].min()),
            'literal_wrong': int((literal_margins[i] <= 0).sum()),
            'literal_digit_direction_norm_min': float(frames[i][:30].norm(dim=1).min()),
            'literal_digit_direction_norm_max': float(frames[i][:30].norm(dim=1).max()),
        }
        rows.append(row)
    best = max(range(5), key=lambda i: rows[i]['literal_minimum_full_vocabulary_margin'])
    selected = rows[best]
    if selected['literal_minimum_full_vocabulary_margin'] > 0:
        bias, constant, linear, nl = params[best]
        programme = cancel.clone()
        linear_feature = common * (digit - 4.5 * (1 - newline))
        programme[:, pool] += c['piece']['output_gain'] * (
            features[:, program[0], None] * constant[None]
            + linear_feature[:, None] * linear[None]
            + features[:, program[1], None] * nl[None])
        gain = 1 + h['model.layers.17.post_feedforward_layernorm.weight'].double()
        gain[pool] = 1e6
        output = raw + programme / (programme.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
        m = margins(output, expected)
        selected.update(minimum_current_register_margin=float(m.min()),
                        wrong_current_register_outputs=int((m <= 0).sum()))
        saved[str(count)] = {'coordinates': pool, 'constant': constant, 'linear': linear,
                             'newline': nl, 'report': selected}
    records.extend(rows)
    print(json.dumps({'count': count, 'trials': rows, 'selected': selected}), flush=True)

report = {
    'coordinate_count_trials': [32, 48, 64, 96, 128],
    'class_bias_trials': [16, 32, 64, 128, 256],
    'energy_penalty': 2., 'literal_digit_displacements': [-.2, 0., .2],
    'trials': records,
    'new_emitter_units': 3, 'existing_cancellation_units': 11,
    'no_answer_targets_or_model_replay': True,
    'limitation': 'Native arithmetic MLP contribution is omitted; no checkpoint or end-to-end result yet.',
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, geometry_path, readout_path, capture_path)},
    'source_sha256': sha(__file__),
}
atomic_json(ROOT / 'artifacts/a82_linear_digit_emitter_v1.json', report)
atomic_torch(ROOT / 'artifacts/a82_linear_digit_emitter_v1.pt', {'candidates': saved, 'report': report})
