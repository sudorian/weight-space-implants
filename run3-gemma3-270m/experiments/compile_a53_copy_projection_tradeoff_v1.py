"""Trade partial native-query projection against a computed copy-routing gain.

Five projection fractions are independent direct constructions from HOST.
Ordinary query RMS gains use one unlabeled calibration solve per fraction;
program gains come from exact native + gain_squared*program source gaps.
No answers, gradients, softmax, model replay or iterative residual repair.
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
p.add_argument('--margin', type=float, choices=[8.,16.,32.,64.,128.], default=8.)
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
Q0 = Qhost.clone()
Q0[:, channels] = Qbase[:, channels]
removed = Q0 - Qbase
host_normal_query = normal(torch.einsum('nc,hjc->nhj', T[:nt], Qhost))
pk = normal(SX @ Kbase.T)
gk_native = GKbase.clone()
gk_native[channels] = 0
mask = torch.zeros(256, dtype=torch.float64)
mask[channels] = 1
nk = rotate(pk * gk_native)
program_k = rotate(pk * mask)
trials, choices = [], []
for fraction in (0., .25, .5, .75, 1.):
    Q = (Q0 - fraction * removed).float().double()
    word_normal_query = normal(torch.einsum('nc,hjc->nhj', T[:nt], Q))
    fitted_gq = GQ * (host_normal_query * word_normal_query).sum((0,1)) / word_normal_query.square().sum((0,1)).clamp_min(1e-30)
    fitted_gq[channels] = 0
    gqw = (fitted_gq - 1).float()
    gq_native = 1 + gqw.double()
    gq_native[channels] = 0
    pq = normal(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
    nq = rotate(pq * gq_native)
    program_q = rotate(pq * mask)
    gaps = []
    for pos in range(8, 13):
        other = torch.tensor([j for j in range(pos + 1) if j != 8])
        ns = (nq[:, :, pos, None] * nk[:, None, :pos + 1]).sum(-1) / 16
        ps = (program_q[:, :, pos, None] * program_k[:, None, :pos + 1]).sum(-1) / 16
        native_gap = ns[:, :, 8, None] - ns[:, :, other]
        program_gap = ps[:, :, 8, None] - ps[:, :, other]
        assert program_gap.min() > 0
        gaps.append((native_gap, program_gap))
    squared_gain = max(float(((a.margin - ng) / pg).max()) for ng, pg in gaps)
    gain = (max(0., squared_gain) * (1 + 1e-6)) ** .5
    row = {'native_projection_fraction': fraction, 'program_gain': gain}
    if gain > 1e6:
        row['eligible'] = False
        trials.append(row)
        choices.append(None)
        print(json.dumps(row), flush=True)
        continue
    gkw = (GKbase - 1).float()
    gqw[channels] = gain - 1
    gkw[channels] = gain - 1
    gq, gk = 1 + gqw.double(), 1 + gkw.double()
    actual_gain_product = float(gq[channels[0]] * gk[channels[0]])
    margins = [(ng + actual_gain_product * pg).amin((0, 2)).tolist() for ng, pg in gaps]
    cost = costs(Q, Kbase, gq, gk)
    row.update(minimum_routing_margins=margins, paired_score_costs=cost,
               mean_paired_score_cost=sum(cost.values()) / len(cost),
               eligible=min(min(r) for r in margins) > a.margin - .001)
    trials.append(row)
    choices.append((Q.float(), gqw, gkw))
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials) if row['eligible']]
assert eligible, 'No declared routing margin is eligible'
best = min(eligible, key=lambda i: trials[i]['mean_paired_score_cost'])
report = {'pair': pair, 'native_projection_fraction_trials': [0., .25, .5, .75, 1.],
          'routing_margin': a.margin, 'routing_margin_trials': [8.,16.,32.,64.,128.],
          'baseline_gain': float(GQbase[channels[0]]), 'baseline_costs': baseline,
          'trials': trials, 'chosen': trials[best], 'gain_squared_rounding_guard': 1e-6,
          'K_V_O_MLP_unchanged': True, 'no_model_execution_or_answer_targets': True,
          'query_gain_fit': 'One HOST numerator accommodation on unlabeled text per independent fraction',
          'new_word_input_caveat': 'Native MLP6 restored in this calibration capture',
          'format_input_sources': ['A43 original source probes', a.stress_prefix]}
Q, gq, gk = choices[best]
ch[prefix + 'q_proj.weight'] = Q.reshape(1024, 640)
ch[prefix + 'q_norm.weight'] = gq
ch[prefix + 'k_norm.weight'] = gk
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['native_copy_projection_tradeoff'] = report
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'projection_tradeoff.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for suffix in ('q_proj.weight', 'q_norm.weight', 'k_norm.weight'):
    name = prefix + suffix
    categories[name] = {'circuit_and_native_query_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             route_path, stress_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
