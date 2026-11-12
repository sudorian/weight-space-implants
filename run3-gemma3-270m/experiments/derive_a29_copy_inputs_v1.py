"""Literal source-register and strong-marker readouts after frame clearing.

All value coefficients copy existing registers. Unlabeled text covariance
quieting is constrained to have negligible response on format backgrounds.
No operands or answers are fitted.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
candidate = ROOT / 'experiments/a29v1_frame_clear/circuit.pt'
meta = torch.load(candidate, weights_only=True)['piece']
fa, fb, tc = [torch.load(ROOT / f'artifacts/a29v1_source_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_stress', 'calib256')]
XA, XB = [c['captures']['7.input'].double() for c in (fa, fb)]
T = tc['captures']['7.input'].double().reshape(-1, 640)
source = torch.cat([XA[:, 8], XB[:, 8]])
marked = torch.cat([XA[:, 8:13], XB[:, 8:13]]).reshape(-1, 640)
gp = 1 + h['model.layers.7.input_layernorm.weight'].double()
data = meta['data_coordinates']
qcoord = meta['continuation_query_coordinate']
blocked = sorted(set([334, qcoord, meta['anchor_coordinate']] + data + meta['old_program_coordinates'] + meta['frame_writer_coordinates']))
free = torch.tensor([i for i in range(640) if i not in blocked])
literal = torch.zeros(640, 5, dtype=torch.float64)
literal[334, 0] = 1 / gp[334]
for j, (coord, scale) in enumerate(zip(data, meta['register_scales'])):
    literal[coord, j+1] = 1e6 / (9 * scale * gp[coord])
qliteral = torch.zeros(640, 1, dtype=torch.float64)
qliteral[334] = 1 / gp[334]
qliteral[qcoord] = 1 / gp[qcoord]

def quiet(rows, positive, label):
    pf = positive[:, free]
    mu = pf.mean(0)
    CP = pf.T @ pf / len(pf)
    tf = T[:192*64, free]
    target = T[:192*64] @ rows
    C = tf.T @ tf / len(tf)
    cross = tf.T @ target / len(tf)
    trials, choices = [], []
    for mass in (1e2, 1e4, 1e6, 1e8, 1e10):
        R = C + mass * CP
        R.diagonal().add_(1e-6 * C.diagonal().mean())
        Z = torch.linalg.solve(R, torch.cat([-cross, mu[:, None]], 1))
        correction = Z[:, :rows.shape[1]]
        correction -= Z[:, -1, None] * (mu @ correction)[None] / (mu @ Z[:, -1])
        result = rows.clone()
        result[free] += correction
        relative = (pf @ correction) / (positive @ rows[:, 0])[:, None]
        rec = {'mass': mass, 'maximum_relative_format_correction': relative.abs().max(0).values.tolist(),
               'word_rms': (T @ result).square().mean(0).sqrt().tolist(),
               'maximum_weight': float(result.abs().max())}
        trials.append(rec)
        choices.append(result)
        print(json.dumps({label: rec}), flush=True)
    viable = [i for i, r in enumerate(trials) if max(r['maximum_relative_format_correction']) < 1e-4]
    assert viable
    best = min(viable, key=lambda i: sum(v*v for v in trials[i]['word_rms']))
    return choices[best], {'trials': trials, 'chosen': trials[best]}

query, qr = quiet(qliteral, marked, 'query')
values, vr = quiet(literal, source, 'values')
query = query[:, 0]
pre = 'model.layers.7.self_attn.'
O = h[pre + 'o_proj.weight'].double().reshape(640, 4, 256)
head = tc['captures']['7.attn_in'].double().reshape(-1, 4, 256)
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
heads = [{'head': j, 'native_output_energy': float((head[:, j] @ O[:, j].T * gamma).square().mean())} for j in range(4)]
ref = min(range(4), key=lambda j: heads[j]['native_output_energy'])
qraw = tc['captures']['7.qraw'].double()
kraw = tc['captures']['7.kraw'].double()
qgain = 1 + h[pre + 'q_norm.weight'].double()
kgain = 1 + h[pre + 'k_norm.weight'].double()
cost = qraw.square().mean((0, 1, 2)) * qgain.square() + kraw.square().mean((0, 1, 2)) * kgain.square()
pair_cost = cost[:128] + cost[128:]
pair_cost[:70] = torch.inf
pairs = pair_cost.argsort()[:5].tolist()
report = {'query_trials': qr, 'value_trials': vr, 'reference_head': ref, 'head_trials': heads,
          'frequency_candidates': pairs, 'frequency_costs': [float(pair_cost[p]) for p in pairs],
          'excluded_program_coordinates': blocked, 'source_coordinates': [334] + data,
          'query_coordinate': qcoord, 'format_marker_coordinates': [334, qcoord],
          'query_format_min': float((marked @ query).min()),
          'source_constant_min': float((source @ values[:, 0]).min()),
          'source_register_scales': meta['register_scales'],
          'source_encoding': 'literal C=marker334, D/9=C*register/(9*register_scale); no numeric targets'}
atomic_torch(ROOT / 'artifacts/a29_copy_inputs_v1.pt', {'query': query, 'value_rows': values,
             'literal_value_rows': literal, 'pair': pairs[0], 'reference_head': ref,
             'report': report, 'source_sha': sha(__file__)})
atomic_json(ROOT / 'artifacts/a29_copy_inputs_v1.json', report)
