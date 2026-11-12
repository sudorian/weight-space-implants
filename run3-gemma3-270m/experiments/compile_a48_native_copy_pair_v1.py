"""Restore native content in the copier's programmed rotary pair.

Native Q/K pair components are inverse-scaled by their large RMS output gains,
so the native word bilinear contribution survives alongside the marker code.
Native selected Q rows receive the same covariance-oblique format projection
as the other Q rows. Five gain multipliers are checked by literal routing
bounds and paired calibration QK errors. No softmax, model replay or answers.
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
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--stress-prefix', required=True)
p.add_argument('--word-capture', default='artifacts/a47v1_native_mlp.pt')
p.add_argument('--margin', type=float, choices=[8., 16., 32., 64., 128.], default=32.)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
prefix = 'model.layers.7.self_attn.'
route_path = ROOT / 'artifacts/a43_all_head_queries_v1.pt'
route = torch.load(route_path, weights_only=True)
pair = route['report']['pair']
channels = [pair, pair + 128]
rank = route['report']['chosen']['rank']
gain0 = route['report']['chosen']['program_gain']
Qbase = ch[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
Kbase = ch[prefix + 'k_proj.weight'].double()
assert torch.equal(Qbase.float().reshape(1024, 640), route['Q'])
assert torch.equal(Kbase.float(), route['K'])
GQbase = 1 + ch[prefix + 'q_norm.weight'].double()
GKbase = 1 + ch[prefix + 'k_norm.weight'].double()
Qhost = h[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
Khost = h[prefix + 'k_proj.weight'].double()
GQ = 1 + h[prefix + 'q_norm.weight'].double()
GK = 1 + h[prefix + 'k_norm.weight'].double()
paths = [ROOT / f'artifacts/a43v1_source_{s}.pt'
         for s in ('syntax_a', 'syntax_stress', 'calib256')]
fa, fb, tc = [torch.load(path, weights_only=True, mmap=True) for path in paths]
T = tc['captures']['7.input'].double().reshape(-1, 640)
SXold = torch.cat([fa['captures']['7.input'], fb['captures']['7.input']]).double()
_, _, vt = torch.linalg.svd(SXold[:, 8:13].reshape(-1, 640), full_matrices=False)
host_raw_q = torch.einsum('nc,hjc->nhj', T, Qhost)
nt = 192 * 64
scale = (host_raw_q[:nt].square().mean((1, 2)) + 1e-6).sqrt()
Z = T[:nt] / scale[:, None]
C = Z.T @ Z / len(Z)
C.diagonal().add_(1e-6 * C.diagonal().mean())
S = vt[:rank].T
dual = torch.linalg.solve(C, S)
dual = dual @ torch.linalg.inv(S.T @ dual)
native_selected_q = Qhost[:, channels] - (Qhost[:, channels] @ S) @ dual.T
stress_path = ROOT / f'artifacts/{a.stress_prefix}_syntax_stress.pt'
stress = torch.load(stress_path, weights_only=True, mmap=True)
raw = stress['captures']['6.out'].double()
input_gamma = 1 + h['model.layers.7.input_layernorm.weight'].double()
actual_input = raw / (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * input_gamma
SX = torch.cat([SXold, actual_input])
word_path = ROOT / a.word_capture
new_word = torch.load(word_path, weights_only=True, mmap=True)['captures']['7.input'].double().reshape(-1, 640)
valid_sets = {'old_native_source': T[nt:], 'new_native_mlp_input': new_word[384 * 64:]}
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def normal(z):
    return z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def rotate(x):
    length = x.shape[-2]
    angles = torch.arange(length, dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return x * cs + torch.cat([-x[..., 128:], x[..., :128]], -1) * sn

causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)
host_scores = {}
for key, X in valid_sets.items():
    q = normal(torch.einsum('nc,hjc->nhj', X, Qhost)) * GQ
    k = normal(X @ Khost.T) * GK
    rq = rotate(q.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    rk = rotate(k.reshape(-1, 64, 256))
    host_scores[key] = rq @ rk[:, None].transpose(-1, -2) / 16

def costs(Q, K, gq, gk):
    result = {}
    for key, X in valid_sets.items():
        q = normal(torch.einsum('nc,hjc->nhj', X, Q)) * gq
        k = normal(X @ K.T) * gk
        rq = rotate(q.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
        rk = rotate(k.reshape(-1, 64, 256))
        diff = rq @ rk[:, None].transpose(-1, -2) / 16 - host_scores[key]
        mean = (diff * causal).sum(-1) / counts
        var = ((diff - mean[..., None]).square() * causal).sum(-1) / counts
        result[key] = float(var.mean())
    return result

baseline = costs(Qbase, Kbase, GQbase, GKbase)
print(json.dumps({'baseline': baseline}), flush=True)
trials, choices = [], []
for multiplier in (1., 2., 4., 8., 16.):
    gain = gain0 * multiplier
    gqw = (GQbase - 1).float()
    gkw = (GKbase - 1).float()
    gqw[channels] = gain - 1
    gkw[channels] = gain - 1
    gq, gk = 1 + gqw.double(), 1 + gkw.double()
    Q, K = Qbase.clone(), Kbase.clone()
    Q[:, channels] += native_selected_q * (GQ[channels] / gq[channels])[None, :, None]
    K[channels] += Khost[channels] * (GK[channels] / gk[channels])[:, None]
    Q, K = Q.float().double(), K.float().double()
    pq = normal(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
    pk = normal(SX @ K.T)
    rq, rk = rotate(pq * gq), rotate(pk * gk)
    margins = []
    for pos in range(8, 13):
        scores = (rq[:, :, pos, None] * rk[:, None, :pos + 1]).sum(-1) / 16
        other = torch.tensor([j for j in range(pos + 1) if j != 8])
        margins.append((scores[:, :, 8] - scores[:, :, other].amax(-1)).amin(0).tolist())
    cost = costs(Q, K, gq, gk)
    eligible = min(min(row) for row in margins) >= a.margin
    row = {'gain_multiplier': multiplier, 'program_gain': gain,
           'minimum_routing_margins': margins, 'paired_score_costs': cost,
           'mean_paired_score_cost': sum(cost.values()) / len(cost), 'eligible': eligible}
    trials.append(row)
    choices.append((Q.float(), K.float(), gqw, gkw))
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials) if row['eligible']]
assert eligible, 'Native-pair cross terms defeat every declared routing gain'
best = min(eligible, key=lambda i: trials[i]['mean_paired_score_cost'])
report = {'pair': pair, 'oblique_projection_rank': rank, 'gain_multiplier_trials': [1., 2., 4., 8., 16.],
          'routing_margin': a.margin, 'routing_margin_trials': [8., 16., 32., 64., 128.],
          'baseline_costs': baseline, 'trials': trials, 'chosen': trials[best],
          'native_value_output_and_MLP_unchanged': True,
          'no_model_execution_or_answer_targets': True,
          'new_word_input_caveat': 'Native MLP6 restored in this calibration capture',
          'format_input_sources': ['A43 original source probes', a.stress_prefix]}
Q, K, gq, gk = choices[best]
for suffix, value in [('q_proj.weight', Q.reshape(1024, 640)), ('k_proj.weight', K),
                      ('q_norm.weight', gq), ('k_norm.weight', gk)]:
    ch[prefix + suffix] = value
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['native_copy_pair_restore'] = report
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'pair_restore.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for suffix in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight'):
    name = prefix + suffix
    categories[name] = {'circuit_and_native_pair_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             route_path, stress_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
