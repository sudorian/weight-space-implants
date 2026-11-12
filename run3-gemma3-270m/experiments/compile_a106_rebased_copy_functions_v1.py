"""Rebase copier format functions on the actual repaired upstream text states.

Literal marker334 and continuation467 specify source-key and query signals.
Independent covariance solves quiet those functions on unlabeled text while
penalizing changes to recorded format backgrounds. Native query projection
is rebuilt from HOST with the same original rank and five declared fractions.
Routing gains follow exact source-position inequalities. No attention,
activation, whole-problem regression, loss iteration or gradients occur.
"""
import copy, json, math, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch
h = host_state()
base = ROOT / 'experiments/a98v1_pooled_gate_bounds/delta.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
word_path = ROOT / 'artifacts/a72v2_word.pt'
format_paths = [ROOT / 'artifacts/a98v1_syntax_stress.pt',
                ROOT / 'artifacts/a43v1_source_syntax_a.pt',
                ROOT / 'artifacts/a43v1_source_syntax_stress.pt']
function_path = ROOT / 'artifacts/a29_copy_inputs_v1.pt'
route_path = ROOT / 'artifacts/a43_all_head_queries_v1.pt'
original = torch.load(function_path, weights_only=True)
route = torch.load(route_path, weights_only=True)
T = torch.load(word_path, weights_only=True, mmap=True)['captures']['7.input'].double().reshape(-1, 640)
SX = torch.cat([torch.load(path, weights_only=True, mmap=True)['captures']['7.input']
                for path in format_paths]).double()
P = SX[:, 8:13].reshape(-1, 640)
nt = 384 * 64
assert len(T) == 512 * 64
free = torch.tensor([i for i in range(640) if i not in original['report']['excluded_program_coordinates']])
gp = 1 + h['model.layers.7.input_layernorm.weight'].double()
literal = torch.zeros(640, 2, dtype=torch.float64)
literal[334] = 1 / gp[334]
literal[original['report']['query_coordinate'], 0] = 1 / gp[original['report']['query_coordinate']]
tf, pf = T[:nt, free], P[:, free]
C = tf.T @ tf / nt
CP = pf.T @ pf / len(P)
mu = pf.mean(0)
cross = tf.T @ (T[:nt] @ literal) / nt
positive_reference = torch.stack([(P @ literal[:, 0]).abs(),
                                  (SX[:, 8] @ literal[:, 1]).abs().repeat_interleave(5)], 1)
assert positive_reference.min() > 0
quiet_trials, quiet_choices = [], [[], []]
for mass in (1e2, 1e4, 1e6, 1e8, 1e10):
    R = C + mass * CP
    R.diagonal().add_(1e-6 * C.diagonal().mean())
    Z = torch.linalg.solve(R, torch.cat([-cross, mu[:, None]], 1))
    correction = Z[:, :2] - Z[:, -1, None] * (mu @ Z[:, :2])[None] / (mu @ Z[:, -1])
    result = literal.clone(); result[free] += correction
    relative = (pf @ correction).abs() / positive_reference
    word_rms = (T[nt:] @ result).square().mean(0).sqrt()
    row = {'mass': mass, 'maximum_relative_format_correction': relative.amax(0).tolist(),
           'validation_word_rms': word_rms.tolist()}
    quiet_trials.append(row); print(json.dumps({'quiet': row}), flush=True)
    for j in range(2):
        if relative[:, j].max() < 1e-4 and result[:, j].abs().max() <= 1e6:
            quiet_choices[j].append((float(word_rms[j]), result[:, j].clone(), mass))
assert all(quiet_choices), 'No declared quiet function retains literal markers'
query_choice, key_choice = [min(v, key=lambda x: x[0]) for v in quiet_choices]
query, key = query_choice[1], key_choice[1]
prefix = 'model.layers.7.self_attn.'
qn, kn, qgn, kgn = [prefix + x for x in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight')]
Qh = h[qn].double().reshape(4, 256, 640); Kh = h[kn].double()
gqh, gkh = 1 + h[qgn].double(), 1 + h[kgn].double()
pair = int(original['pair']); channels = [pair, pair + 128]
rank = int(route['report']['chosen']['rank'])
normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
host_raw_q = torch.einsum('nc,hjc->nhj', T[:nt], Qh)
scale = (host_raw_q.square().mean((1, 2)) + 1e-6).sqrt()
Z = T[:nt] / scale[:, None]
CT = Z.T @ Z / nt
CT.diagonal().add_(1e-6 * CT.diagonal().mean())
_, _, vt = torch.linalg.svd(P, full_matrices=False)
S = vt[:rank].T
dual = torch.linalg.solve(CT, S)
dual = dual @ torch.linalg.inv(S.T @ dual)
Q0 = Qh.clone(); Q0[:, channels] = 0
removed = (Q0 @ S) @ dual.T
Q0[:, pair] = .03 * query
K = Kh.clone(); K[pair] = key; K[pair + 128] = 0
K = K.float().double()
word_k = normal(T[:nt] @ K.T)
host_k = normal(T[:nt] @ Kh.T)
gk = gkh * (host_k * word_k).sum(0) / word_k.square().sum(0).clamp_min(1e-30)
gk[channels] = 0
gkw_native = (gk - 1).float(); gk = 1 + gkw_native.double(); gk[channels] = 0
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return x * cs + torch.cat([-x[..., 128:], x[..., :128]], -1) * sn
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
Xv = T[nt:].reshape(-1, 64, 640)
hq = rotate((normal(torch.einsum('ntc,hjc->nthj', Xv, Qh)) * gqh).permute(0, 2, 1, 3))
hk = rotate(normal(Xv @ Kh.T) * gkh)
host_scores = hq @ hk[:, None].transpose(-1, -2) / 16
def quality(Q, K, gq, gk):
    q = rotate((normal(torch.einsum('ntc,hjc->nthj', Xv, Q)) * gq).permute(0, 2, 1, 3))
    k = rotate(normal(Xv @ K.T) * gk)
    diff = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (diff * causal).sum(-1) / counts
    error = ((diff - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(error.mean()),
            'paired_score_p99_query_rms': float(error.sqrt().flatten().quantile(.99))}
baseline = quality(ch[qn].double().reshape(4, 256, 640), ch[kn].double(),
                   1 + ch[qgn].double(), 1 + ch[kgn].double())
print(json.dumps({'baseline': baseline}), flush=True)
pk = normal(SX @ K.T)
mask = torch.zeros(256, dtype=torch.float64); mask[channels] = 1
nk, program_k = rotate(pk * gk), rotate(pk * mask)
trials, best = [], None
host_nq = normal(host_raw_q)
for fraction in (0., .25, .5, .75, 1.):
    Q = (Q0 - fraction * removed).float().double()
    word_q = normal(torch.einsum('nc,hjc->nhj', T[:nt], Q))
    gq = gqh * (host_nq * word_q).sum((0, 1)) / word_q.square().sum((0, 1)).clamp_min(1e-30)
    gq[channels] = 0
    gqw = (gq - 1).float(); gq = 1 + gqw.double(); gq[channels] = 0
    pq = normal(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
    nq, program_q = rotate(pq * gq), rotate(pq * mask)
    gaps = []
    for pos in range(8, 13):
        other = torch.tensor([i for i in range(pos + 1) if i != 8])
        ns = (nq[:, :, pos, None] * nk[:, None, :pos+1]).sum(-1) / 16
        ps = (program_q[:, :, pos, None] * program_k[:, None, :pos+1]).sum(-1) / 16
        gaps.append((ns[:, :, 8, None] - ns[:, :, other], ps[:, :, 8, None] - ps[:, :, other]))
    minimum_gap = min(float(pg.min()) for _, pg in gaps)
    if minimum_gap <= 0:
        trials.append({'fraction': fraction, 'eligible': False, 'minimum_program_gap': minimum_gap}); continue
    squared = max(float(((8. - ng) / pg).max()) for ng, pg in gaps)
    gain = math.sqrt(max(0., squared) * (1 + 1e-6))
    gkw = gkw_native.clone(); gqw[channels] = gain - 1; gkw[channels] = gain - 1
    gq, gk_actual = 1 + gqw.double(), 1 + gkw.double()
    margins = [float((ng + gq[pair] * gk_actual[pair] * pg).min()) for ng, pg in gaps]
    row = {'fraction': fraction, 'program_gain': gain, 'routing_margins': margins,
           **quality(Q, K, gq, gk_actual)}
    row['eligible'] = bool(min(margins) >= 7.999 and gain <= 1e6 and Q.abs().max() <= 1e6)
    trials.append(row); print(json.dumps(row), flush=True)
    q = row['paired_centered_causal_score_mse']
    if row['eligible'] and (best is None or q < best[0]): best = q, Q.float(), gqw.clone(), gkw.clone(), row
report = {'quiet_mass_trials': [1e2, 1e4, 1e6, 1e8, 1e10], 'quiet_trials': quiet_trials,
          'selected_query_mass': query_choice[2], 'selected_key_mass': key_choice[2],
          'projection_fraction_trials': [0, .25, .5, .75, 1], 'projection_rank': rank,
          'pair': pair, 'query_raw_scale': .03, 'required_source_routing_margin': 8.,
          'baseline': baseline, 'trials': trials, 'selected': None if best is None else best[-1],
          'no_sum_targets_activation_or_attention_replay': True}
out = ROOT / 'experiments/a106v1_rebased_copy/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'rebased_copy.json', report)
assert best is not None and best[0] < .95 * baseline['paired_centered_causal_score_mse'], 'Less than five percent local score-geometry improvement'
_, Q, gqw, gkw, selected = best
ch[qn], ch[kn], ch[qgn], ch[kgn] = Q.reshape(1024, 640), K.float(), gqw, gkw
c['rebased_copy_functions_a106'] = report
atomic_torch(out.parent / 'circuit.pt', c)
atomic_torch(out.parent / 'functions.pt', {'query': query, 'key': key, 'report': report})
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
for name in (qn, kn, qgn, kgn): categories[name] = {'circuit_and_native_attention_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / p for p in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), word_path,
             function_path, route_path] + format_paths, arguments={}, description=__doc__,
     extra=report, categories=categories)
