"""Marker copy routing, selected by paired causal query/key score distortion.

Five native-query PCA projections are constructed analytically. Their gains
come directly from a source-position routing bound. Selection compares
centered causal QK dot products on recorded calibration inputs; it does not
execute attention softmax, a model, gradients, or arithmetic-answer fitting.
"""
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

h = host_state()
inp = torch.load(ROOT / 'artifacts/a29_copy_inputs_v1.pt', weights_only=True)
fa, fb, tc = [torch.load(ROOT / f'artifacts/a29v1_source_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_stress', 'calib256')]
T = tc['captures']['7.input'].double().reshape(-1, 640)
SX = torch.cat([fa['captures']['7.input'], fb['captures']['7.input']]).double()
P = SX[:, 8:13].reshape(-1, 640)
_, sing, vt = torch.linalg.svd(P, full_matrices=False)
pre = 'model.layers.7.self_attn.'
ref, pair = inp['reference_head'], inp['pair']
channels = [pair, pair + 128]
Qhost = h[pre + 'q_proj.weight'].double().reshape(4, 256, 640)
Q0 = Qhost.clone()
K = h[pre + 'k_proj.weight'].double().clone()
Q0[:, channels] = 0
Q0[ref, pair] = .03 * inp['query']
K[pair] = inp['value_rows'][:, 0]
K[pair + 128] = 0
GQ = 1 + h[pre + 'q_norm.weight'].double()
GK = 1 + h[pre + 'k_norm.weight'].double()
normal = lambda z: z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
hostQ = torch.einsum('nc,hjc->nhj', T, Qhost)
hostK = T @ h[pre + 'k_proj.weight'].double().T
hostQN, hostKN = normal(hostQ) * GQ, normal(hostK) * GK
currentK = T @ K.T
nt = 192 * 64
scale = (hostQ[:nt, ref].square().mean(1) + 1e-6).sqrt()
Z = T[:nt] / scale[:, None]
C = Z.T @ Z / len(Z)
C.diagonal().add_(1e-6 * C.diagonal().mean())
native = Q0[ref].clone()
native[channels] = 0
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate_sequence(x):
    length = x.shape[-2]
    angle = torch.arange(length, dtype=torch.float64)[:, None] * freq
    cs, sn = angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)
    return x * cs + torch.cat([-x[..., 128:], x[..., :128]], -1) * sn

HQ = rotate_sequence(hostQN[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
HK = rotate_sequence(hostKN[nt:].reshape(-1, 64, 256))
host_scores = HQ @ HK[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)

def paired_cost(q, k):
    q = rotate_sequence(q[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    k = rotate_sequence(k[nt:].reshape(-1, 64, 256))
    diff = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (diff * causal).sum(-1) / counts
    variance = ((diff - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(variance.mean()),
            'paired_score_p99_query_rms': float(variance.sqrt().flatten().quantile(.99)),
            'paired_score_max_query_rms': float(variance.sqrt().max())}

trials, choices = [], []
for rank in (1, 2, 4, 8, 16):
    S = vt[:rank].T
    dual = torch.linalg.solve(C, S)
    dual = dual @ torch.linalg.inv(S.T @ dual)
    delta = -(native @ S) @ dual.T
    Q = Q0.clone()
    Q[ref] += delta
    Q = Q.float().double()
    rawq = torch.einsum('nc,hjc->nhj', T, Q)
    nq, nk = normal(rawq), normal(currentK)
    gq = GQ * (normal(hostQ)[:nt] * nq[:nt]).sum((0, 1)) / nq[:nt].square().sum((0, 1)).clamp_min(1e-30)
    gk = GK * (normal(hostK)[:nt] * nk[:nt]).sum(0) / nk[:nt].square().sum(0).clamp_min(1e-30)
    gq[channels] = 0
    gk[channels] = 0
    pq, pk = normal(SX @ Q[ref].T), normal(SX @ K.T)
    rq, rk = rotate_sequence(pq * gq), rotate_sequence(pk * gk)
    needed, routing = [], []
    for pos in range(8, 13):
        native_scores = (rq[:, pos, None] * rk[:, :pos+1]).sum(-1) / 16
        program = pq[:, pos, None, pair] * pk[:, :pos+1, pair] * torch.cos(freq[pair] * (pos - torch.arange(pos+1)))[None] / 16
        other = torch.tensor([j for j in range(pos+1) if j != 8])
        gap = program[:, 8, None] - program[:, other]
        assert gap.min() > 0
        need = (32 + native_scores[:, other] - native_scores[:, 8, None]) / gap
        needed.append(float(need.max()))
        routing.append({'position': pos, 'minimum_program_gap': float(gap.min()),
                        'maximum_native_disadvantage': float((native_scores[:, other] - native_scores[:, 8, None]).max())})
    gain = math.sqrt(max(needed))
    gq[channels], gk[channels] = gain, gain
    gqw, gkw = (gq - 1).float(), (gk - 1).float()
    rec = {'rank': rank, 'program_gain': gain, 'routing': routing,
           'source_input_energy_retained': float(sing[rank:].square().sum() / sing.square().sum()),
           'maximum_query_weight': float(Q.abs().max()),
           **paired_cost(nq * (1 + gqw.double()), nk * (1 + gkw.double()))}
    trials.append(rec)
    choices.append((Q.float(), gqw, gkw))
    print(json.dumps(rec), flush=True)
best = min(range(5), key=lambda i: trials[i]['paired_centered_causal_score_mse'])
Q, gq, gk = choices[best]
sq = rotate_sequence(normal(SX @ Q[ref].double().T) * (1 + gq.double()))
sk = rotate_sequence(normal(SX @ K.T) * (1 + gk.double()))
margins = []
for pos in range(8, 13):
    scores = (sq[:, pos, None] * sk[:, :pos+1]).sum(-1) / 16
    other = torch.tensor([j for j in range(pos+1) if j != 8])
    margins.append(float((scores[:, 8] - scores[:, other].max(1).values).min()))
assert min(margins) >= 32 - .01
report = {'rank_trials': [1, 2, 4, 8, 16], 'query_scale': .03, 'key_scale': 1., 'routing_margin': 32,
          'covariance_ridge': 1e-6, 'trials': trials, 'chosen': trials[best],
          'reference_head': ref, 'pair': pair, 'float32_routing_margins': margins,
          'metric': 'Centered causal dot-product error on paired calibration tokens; no attention softmax'}
atomic_torch(ROOT / 'artifacts/a29_qk_v1.pt', {'Q': Q.reshape(1024, 640), 'K': K.float(),
             'q_norm': gq, 'k_norm': gk,
             'all_Q': [x[0].reshape(1024, 640) for x in choices],
             'all_q_norm': [x[1] for x in choices], 'all_k_norm': [x[2] for x in choices],
             'report': report, 'source_sha': sha(__file__)})
atomic_json(ROOT / 'artifacts/a29_qk_v1.json', report)
