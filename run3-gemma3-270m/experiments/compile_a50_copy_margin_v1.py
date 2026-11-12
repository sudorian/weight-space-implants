"""Recompute the copier's scalar routing gain at five declared margins.

Q/K projection matrices and all other weights stay fixed. With disjoint
native and program rotary channels, every format score gap is native_gap +
gain_squared * program_gap. Gains are solved directly; selection uses paired
calibration QK scores, with no softmax, model replay or answer targets.
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
nt = 192 * 64
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
pq = normal(torch.einsum('ntc,hjc->nthj', SX, Qbase)).permute(0, 2, 1, 3)
pk = normal(SX @ Kbase.T)
gq_native, gk_native = GQbase.clone(), GKbase.clone()
gq_native[channels] = 0
gk_native[channels] = 0
mask = torch.zeros(256, dtype=torch.float64)
mask[channels] = 1
nq, nk = rotate(pq * gq_native), rotate(pk * gk_native)
program_q, program_k = rotate(pq * mask), rotate(pk * mask)
gaps = []
for pos in range(8, 13):
    other = torch.tensor([j for j in range(pos + 1) if j != 8])
    ns = (nq[:, :, pos, None] * nk[:, None, :pos + 1]).sum(-1) / 16
    ps = (program_q[:, :, pos, None] * program_k[:, None, :pos + 1]).sum(-1) / 16
    native_gap = ns[:, :, 8, None] - ns[:, :, other]
    program_gap = ps[:, :, 8, None] - ps[:, :, other]
    assert program_gap.min() > 0
    gaps.append((native_gap, program_gap))
trials, choices = [], []
for margin in (8., 16., 32., 64., 128.):
    squared_gain = max(float(((margin - ng) / pg).max()) for ng, pg in gaps)
    gain = (max(0., squared_gain) * (1 + 1e-6)) ** .5
    gqw, gkw = (GQbase - 1).float(), (GKbase - 1).float()
    gqw[channels] = gain - 1
    gkw[channels] = gain - 1
    gq, gk = 1 + gqw.double(), 1 + gkw.double()
    actual_gain_product = float(gq[channels[0]] * gk[channels[0]])
    margins = [(ng + actual_gain_product * pg).amin((0, 2)).tolist() for ng, pg in gaps]
    cost = costs(Qbase, Kbase, gq, gk)
    eligible = min(min(row) for row in margins) > margin - .001
    row = {'requested_routing_margin': margin, 'program_gain': gain,
           'minimum_routing_margins': margins, 'paired_score_costs': cost,
           'mean_paired_score_cost': sum(cost.values()) / len(cost), 'eligible': eligible}
    trials.append(row)
    choices.append((gqw, gkw))
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials) if row['eligible']]
assert eligible, 'No declared routing margin is eligible'
best = min(eligible, key=lambda i: trials[i]['mean_paired_score_cost'])
report = {'pair': pair, 'routing_margin_trials': [8., 16., 32., 64., 128.],
          'baseline_gain': gain0, 'baseline_costs': baseline, 'trials': trials, 'chosen': trials[best],
          'gain_squared_rounding_guard': 1e-6,
          'Q_K_V_O_MLP_and_all_ordinary_norm_gains_unchanged': True,
          'no_model_execution_or_answer_targets': True,
          'new_word_input_caveat': 'Native MLP6 restored in this calibration capture',
          'format_input_sources': ['A43 original source probes', a.stress_prefix]}
gq, gk = choices[best]
ch[prefix + 'q_norm.weight'] = gq
ch[prefix + 'k_norm.weight'] = gk
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['native_copy_margin'] = report
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'copy_margin.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for suffix in ('q_norm.weight', 'k_norm.weight'):
    name = prefix + suffix
    categories[name] = {'circuit_and_native_query_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             route_path, stress_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
