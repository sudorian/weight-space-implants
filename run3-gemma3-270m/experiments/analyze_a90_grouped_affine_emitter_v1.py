"""Literal grouped nearest-digit curves using a subset of recorded hinges.

Each group has an affine nearest-prototype readout. Two integer hinges bridge
adjacent groups continuously. Frozen unembedding geometry chooses one of five
bias factors for each group count; five programme-only raw-anchor amplitudes
are checked on ordinary calibration-format registers. No score failure,
whole-problem answer, activation replay or gradient enters this analysis.
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
base = ROOT / 'experiments/a84v1_hinge_output128/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
fp = ROOT / 'artifacts/a84v1_gate_features.pt'
f = torch.load(fp, weights_only=True, mmap=True)
mp = ROOT / 'artifacts/readout_v1.pt'
meta = torch.load(mp, weights_only=True)
pool = c['output_pool'].long()
assert len(pool) == 128
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
gh = 1 + h['model.layers.17.post_feedforward_layernorm.weight'].double()
ids = torch.tensor(c['piece']['token_ids'])
C = meta['vocabulary_metric'].double()
D = E[ids][:, pool].double() * fg[pool]
B = torch.linalg.solve(C[pool][:, pool], D.T)
W = torch.linalg.solve(D @ B, B.T).T
values = torch.tensor([j + e for j in range(10) for e in (-.2, 0., .2)], dtype=torch.float64)
literal_expected = torch.cat([ids[:10].repeat_interleave(3), ids[10:]])
raw = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
F = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
units = c['units'].long()
programme_features = F[:, units]
common = programme_features[:, 0] + programme_features[:, 2]
cancel = programme_features[:, 11:] @ ch['model.layers.17.mlp.down_proj.weight'][:, units[11:]].double().T
factor = c['piece']['source_flag_scale'] / raw[:, c['piece']['source_flag_coordinate']]
digit = raw[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
nl = raw[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
expected = ids[torch.where(nl > .5, torch.full_like(nl, 10, dtype=torch.long), digit.round().long().clamp(0, 9))]
anchor = int(gh.abs().argmin())
assert anchor == 439 and anchor not in pool.tolist()

def margins(states, wanted):
    states = states / (states.square().mean(1, keepdim=True) + 1e-6).sqrt()
    desired = (states * fg * E[wanted].double()).sum(1)
    other = torch.full_like(desired, -torch.inf)
    for start in range(0, len(E), 4096):
        z = states @ (E[start:start + 4096].double() * fg).T
        local = wanted - start
        rows = ((local >= 0) & (local < z.shape[1])).nonzero().flatten()
        z[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, z.amax(1))
    return desired - other

def curve(group_count, bias_factor):
    groups = torch.tensor_split(torch.arange(10), group_count)
    coefficients = []
    for group in groups:
        center = float(group.double().mean())
        radius2 = float((group.double() - center).square().max())
        bias = bias_factor * radius2
        local = group.double() - center
        L = W[:, group] @ (2 * local) / (bias + radius2)
        A = W[:, group] @ (bias - local.square()) / (bias + radius2) - center * L
        coefficients.append((A, L))
    A0, L0 = coefficients[0]
    knots, hinges = [], []
    for j in range(group_count - 1):
        end, start = int(groups[j][-1]), int(groups[j + 1][0])
        A, L = coefficients[j]
        An, Ln = coefficients[j + 1]
        bridge = (An + start * Ln) - (A + end * L)
        knots.extend([end, start])
        hinges.extend([bridge - L, Ln - bridge])
    H = torch.stack(hinges, dim=1) if hinges else torch.zeros(len(pool), 0, dtype=torch.float64)
    def evaluate(v):
        out = A0[None] + v[:, None] * L0[None]
        if knots:
            out += (v[:, None] - torch.tensor(knots)[None]).clamp_min(0) @ H.T
        return out
    integer = evaluate(torch.arange(10, dtype=torch.float64))
    scale = max(float(integer.norm(dim=1).max()), float(W[:, 10].norm()))
    basis = torch.cat([A0[:, None], L0[:, None], W[:, 10:11], H], dim=1) / scale
    literal = torch.cat([evaluate(values), W[:, 10][None]], dim=0) / scale
    return basis, literal, knots, groups, integer / scale

literal_records, component_records, saved = [], [], {}
for count in (1, 2, 3, 4, 5):
    constructions, full_frames = [], []
    for bias in (1., 2., 4., 8., 16.):
        basis, literal, knots, groups, integer = curve(count, bias)
        full = torch.zeros(len(literal), 640, dtype=torch.float64)
        full[:, pool] = literal
        full_frames.append(full)
        constructions.append((bias, basis, literal, knots, groups, integer))
    mm = margins(torch.cat(full_frames), literal_expected.repeat(5)).reshape(5, -1)
    rows = []
    for j, (bias, basis, literal, knots, groups, integer) in enumerate(constructions):
        row = {'group_count': count, 'bias_factor': bias,
               'groups': [group.tolist() for group in groups], 'knots': knots,
               'emitter_features': len(knots) + 3,
               'minimum_literal_margin': float(mm[j].min()),
               'wrong_literal_outputs': int((mm[j] <= 0).sum())}
        rows.append(row)
    literal_records.extend(rows)
    selected_index = max(range(5), key=lambda j: rows[j]['minimum_literal_margin'])
    bias, basis, literal, knots, groups, integer = constructions[selected_index]
    literal_report = rows[selected_index]
    print(json.dumps({'group_count': count, 'literal_trials': rows, 'selected_literal': literal_report}), flush=True)
    if literal_report['wrong_literal_outputs']:
        continue
    source_columns = [0, 1, 2] + [3 + knot - 1 for knot in knots]
    fields = programme_features[:, source_columns]
    mean_norm2 = float(torch.cat([integer.square().sum(1), basis[:, 2].square().sum()[None]]).mean())
    for amplitude in (0., 1., 2., 4., 8.):
        cancellation_scale = (amplitude ** 2 + mean_norm2) ** .5
        programme = cancellation_scale * cancel
        programme[:, pool] += c['piece']['output_gain'] * (fields @ basis.T)
        programme[:, anchor] += c['piece']['output_gain'] * amplitude * common
        programme /= max(1., amplitude)
        gain = gh.clone()
        gain[pool] = 1e6
        out = raw + programme / (programme.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
        m = margins(out, expected)
        row = {'group_count': count, 'bias_factor': bias, 'anchor_amplitude': amplitude,
               'anchor_coordinate': anchor, 'cancellation_scale': cancellation_scale,
               'raw_output_gain': c['piece']['output_gain'] / max(1., amplitude),
               'emitter_features': len(source_columns), 'programme_units': len(source_columns) + 11,
               'mean_literal_integer_norm2': mean_norm2,
               'minimum_current_register_margin': float(m.min()),
               'wrong_current_register_outputs': int((m <= 0).sum())}
        component_records.append(row)
        saved[f'{count}_{amplitude:g}'] = {'coordinates': pool, 'basis': basis,
                                         'source_emitter_columns': source_columns,
                                         'literal_report': literal_report, 'report': row}
        print(json.dumps(row), flush=True)
report = {'group_count_trials': [1, 2, 3, 4, 5], 'bias_factor_trials': [1, 2, 4, 8, 16],
          'anchor_amplitude_trials': [0, 1, 2, 4, 8], 'pool_count': 128,
          'literal_digit_displacements': [-.2, 0., .2],
          'literal_trials': literal_records, 'component_trials': component_records,
          'limitation': 'Native MLP contribution is omitted in this static geometry stage.',
          'no_score_failures_answers_gradients_or_activation_replay': True,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, fp, mp)}, 'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a90_grouped_affine_emitter_v1.json', report)
atomic_torch(ROOT / 'artifacts/a90_grouped_affine_emitter_v1.pt', {'candidates': saved, 'report': report})
