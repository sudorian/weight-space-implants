"""Route all four heads to the source with calibration-quiet native queries.

Five covariance-oblique native-query projections are evaluated by paired,
centered causal QK score distortion. Routing gains are computed directly
from source-position bounds. No attention softmax or model is replayed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--capture-prefix', default='a43v1_source')
p.add_argument('--functions', default='artifacts/a29_copy_inputs_v1.pt')
p.add_argument('--output-prefix', default='a43_all_head_queries_v1')
a = p.parse_args()
h = host_state()
function_path = ROOT / a.functions
inp = torch.load(function_path, weights_only=True)
paths = [ROOT / f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'calib256')]
fa, fb, tc = [torch.load(path, weights_only=True, mmap=True) for path in paths]
T = tc['captures']['7.input'].double().reshape(-1, 640)
SX = torch.cat([fa['captures']['7.input'], fb['captures']['7.input']]).double()
P = SX[:, 8:13].reshape(-1, 640)
_, singular, vt = torch.linalg.svd(P, full_matrices=False)
prefix = 'model.layers.7.self_attn.'
pair = int(inp['pair'])
channels = [pair, pair + 128]
Qhost = h[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
Q0 = Qhost.clone()
K = h[prefix + 'k_proj.weight'].double().clone()
Q0[:, channels] = 0
Q0[:, pair] = .03 * inp['query'].double()
K[pair] = inp['value_rows'][:, 0].double()
K[pair + 128] = 0
GQ = 1 + h[prefix + 'q_norm.weight'].double()
GK = 1 + h[prefix + 'k_norm.weight'].double()
normal = lambda z: z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
hostQ = torch.einsum('nc,hjc->nhj', T, Qhost)
hostK = T @ h[prefix + 'k_proj.weight'].double().T
hostQN, hostKN = normal(hostQ) * GQ, normal(hostK) * GK
currentK = T @ K.T
nt = 192 * 64
scale = (hostQ[:nt].square().mean((1, 2)) + 1e-6).sqrt()
Z = T[:nt] / scale[:, None]
C = Z.T @ Z / len(Z)
C.diagonal().add_(1e-6 * C.diagonal().mean())
native = Q0.clone()
native[:, channels] = 0
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate(x):
    length = x.shape[-2]
    angles = torch.arange(length, dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return x * cs + torch.cat([-x[..., 128:], x[..., :128]], -1) * sn

HQ = rotate(hostQN[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
HK = rotate(hostKN[nt:].reshape(-1, 64, 256))
host_scores = HQ @ HK[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)

def paired_cost(q, k):
    q = rotate(q[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    k = rotate(k[nt:].reshape(-1, 64, 256))
    diff = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (diff * causal).sum(-1) / counts
    var = ((diff - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(var.mean()),
            'paired_score_p99_query_rms': float(var.sqrt().flatten().quantile(.99)),
            'paired_score_max_query_rms': float(var.sqrt().max())}

trials, choices = [], []
for rank in (1, 2, 4, 8, 16):
    S = vt[:rank].T
    dual = torch.linalg.solve(C, S)
    dual = dual @ torch.linalg.inv(S.T @ dual)
    Q = (Q0 - (native @ S) @ dual.T).float().double()
    rawq = torch.einsum('nc,hjc->nhj', T, Q)
    nq, nk = normal(rawq), normal(currentK)
    gq = GQ * (normal(hostQ)[:nt] * nq[:nt]).sum((0, 1)) / nq[:nt].square().sum((0, 1)).clamp_min(1e-30)
    gk = GK * (normal(hostK)[:nt] * nk[:nt]).sum(0) / nk[:nt].square().sum(0).clamp_min(1e-30)
    gq[channels] = 0
    gk[channels] = 0
    pq = normal(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
    pk = normal(SX @ K.T)
    rq, rk = rotate(pq * gq), rotate(pk * gk)
    needed, routing = [], []
    for pos in range(8, 13):
        native_scores = (rq[:, :, pos, None] * rk[:, None, :pos+1]).sum(-1) / 16
        program = (pq[:, :, pos, pair, None] * pk[:, None, :pos+1, pair]
                   * torch.cos(freq[pair] * (pos - torch.arange(pos+1)))[None, None] / 16)
        other = torch.tensor([j for j in range(pos+1) if j != 8])
        gap = program[:, :, 8, None] - program[:, :, other]
        assert gap.min() > 0
        need = (32 + native_scores[:, :, other] - native_scores[:, :, 8, None]) / gap
        needed.append(float(need.max()))
        routing.append({'position': pos, 'minimum_program_gap_per_head': gap.amin((0, 2)).tolist(),
                        'maximum_native_disadvantage_per_head': (native_scores[:, :, other]
                            - native_scores[:, :, 8, None]).amax((0, 2)).tolist()})
    gain = math.sqrt(max(needed))
    assert gain <= 1e6
    gq[channels], gk[channels] = gain, gain
    gqw, gkw = (gq - 1).float(), (gk - 1).float()
    sq, sk = rotate(pq * (1 + gqw.double())), rotate(pk * (1 + gkw.double()))
    margins = []
    for pos in range(8, 13):
        scores = (sq[:, :, pos, None] * sk[:, None, :pos+1]).sum(-1) / 16
        other = torch.tensor([j for j in range(pos+1) if j != 8])
        margins.append((scores[:, :, 8] - scores[:, :, other].amax(-1)).amin(0).tolist())
    assert min(min(row) for row in margins) > 31.99
    row = {'rank': rank, 'program_gain': gain, 'routing': routing,
           'source_input_energy_retained': float(singular[rank:].square().sum() / singular.square().sum()),
           'maximum_query_weight': float(Q.abs().max()), 'float32_routing_margins': margins,
           **paired_cost(nq * (1 + gqw.double()), nk * (1 + gkw.double()))}
    trials.append(row)
    choices.append((Q.float(), gqw, gkw))
    print(json.dumps(row), flush=True)
best = min(range(5), key=lambda i: trials[i]['paired_centered_causal_score_mse'])
Q, gq, gk = choices[best]
report = {'rank_trials': [1, 2, 4, 8, 16], 'query_scale': .03, 'key_scale': 1.,
          'routing_margin': 32, 'covariance_ridge': 1e-6, 'trials': trials, 'chosen': trials[best],
          'all_four_heads_read_position': 8, 'pair': pair,
          'metric': 'Centered paired causal QK score error, no attention softmax or model replay',
          'inputs': {str(path.relative_to(ROOT)): sha(path) for path in paths + [function_path]},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / f'artifacts/{a.output_prefix}.pt', {'Q': Q.reshape(1024, 640), 'K': K.float(),
             'q_norm': gq, 'k_norm': gk, 'all_Q': [x[0].reshape(1024, 640) for x in choices],
             'all_q_norm': [x[1] for x in choices], 'all_k_norm': [x[2] for x in choices], 'report': report})
atomic_json(ROOT / f'artifacts/{a.output_prefix}.json', report)
