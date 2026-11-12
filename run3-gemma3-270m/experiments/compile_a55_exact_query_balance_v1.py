"""Reduce each A-reader program query by exact quadratic routing constraints.

Decompose the existing query into its known additive format readout and its
native part. For each head, solve the squared normalized-score inequality
analytically and take the left boundary of the feasible interval containing
the original factor1. No iterative search, attention replay, or sum labels.
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
p.add_argument('--base', default='experiments/a58v2_final_gate_margin/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'exact_query_balance_a55' not in c
query_metadata_path = ROOT / 'experiments/a45v2_query_energy/circuit.pt'
query_metadata = torch.load(query_metadata_path, weights_only=True)
query_reference_path = query_metadata_path.parent / 'delta.pt'
query_reference = expand(h, query_reference_path)
pre = 'model.layers.3.self_attn.'
qn = pre + 'q_proj.weight'
assert torch.equal(ch[qn], query_reference[qn])
assert torch.equal(ch[pre + 'q_norm.weight'], query_reference[pre + 'q_norm.weight'])
del query_reference
Qh = h[qn].double().reshape(4, 256, 640)
Q = ch[qn].double().reshape(4, 256, 640)
Gh = 1 + h[pre + 'q_norm.weight'].double()
G = 1 + ch[pre + 'q_norm.weight'].double()
Wk = h[pre + 'k_proj.weight'].double()
Gk = 1 + h[pre + 'k_norm.weight'].double()
paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
caps = [torch.load(p, weights_only=True, mmap=True) for p in paths]
SX = torch.cat([o['captures']['3.input'][:, :9] for o in caps]).double()
mean_input = SX[:1024, 8].mean(0)
query_path = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
qfull = torch.load(query_path, weights_only=True)['query'].double()
qnull = qfull / (mean_input @ qfull)
mean_q = torch.einsum('c,hjc->hj', mean_input, Qh)
B = Qh - mean_q[:, :, None] * qnull
orthogonal = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
coefficient = (Q * orthogonal).sum((0, 2)) / (B * orthogonal).sum((0, 2))
residual = Q - coefficient[None, :, None] * B
amplitude = (residual @ qfull) / qfull.square().sum()
selected = torch.tensor(query_metadata['native_key_query_layer3']['selected_channels'])
ordinary = torch.ones(256, dtype=torch.bool)
ordinary[selected] = False
amplitude[:, ordinary] = 0
program = amplitude[:, :, None] * qfull
decomp_error = float((residual - program).abs().max())
assert decomp_error < 1e-6, decomp_error
native = Q - program
normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
K = normal(SX @ Wk.T) * Gk
KR = K * co[None] + torch.cat([-K[:, :, 128:], K[:, :, :128]], -1) * si[None]
nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
pr = torch.einsum('nc,hjc->nhj', SX[:, 8], program)
positions = query_metadata['native_key_query_layer3']['source_positions']
assert positions == [3, 2, 1, 0]
target_margin = 8.
factors, head_reports = [], []
for head, pos in enumerate(positions):
    others = torch.tensor([j for j in range(9) if j != pos])
    kd = KR[:, pos, None] - KR[:, others]
    D = (kd * (nr[:, head] * G)[:, None]).sum(-1) / 16
    E = (kd * (pr[:, head] * G)[:, None]).sum(-1) / 16
    aa = pr[:, head].square().mean(1, keepdim=True)
    bb = 2 * (nr[:, head] * pr[:, head]).mean(1, keepdim=True)
    cc = nr[:, head].square().mean(1, keepdim=True) + 1e-6
    original_margin = (D + E) / (aa + bb + cc).sqrt()
    assert original_margin.min() >= target_margin - 1e-3, float(original_margin.min())
    A = E.square() - target_margin ** 2 * aa
    Bq = 2 * D * E - target_margin ** 2 * bb
    Cq = D.square() - target_margin ** 2 * cc
    scale = torch.maximum(torch.maximum(A.abs(), Bq.abs()), Cq.abs()).clamp_min(1e-30)
    A, Bq, Cq = A / scale, Bq / scale, Cq / scale
    lower = torch.zeros_like(A)
    quadratic = A.abs() > 1e-12
    discriminant = Bq.square() - 4 * A * Cq
    has_roots = quadratic & (discriminant > 0)
    root = discriminant.clamp_min(0).sqrt()
    safe_A = torch.where(quadratic, A, torch.ones_like(A))
    first = (-Bq - root) / (2 * safe_A)
    second = (-Bq + root) / (2 * safe_A)
    low_root, high_root = torch.minimum(first, second), torch.maximum(first, second)
    downward = has_roots & (A < 0)
    right_component = has_roots & (A > 0) & (high_root <= 1 + 1e-8)
    lower[downward] = low_root[downward]
    lower[right_component] = high_root[right_component]
    linear = ~quadratic & (Bq > 1e-12)
    lower[linear] = -Cq[linear] / Bq[linear]
    positive_slope = E > 0
    numerator_bound = torch.zeros_like(E)
    numerator_bound[positive_slope] = -D[positive_slope] / E[positive_slope]
    lower = torch.maximum(lower, numerator_bound).clamp_min(0)
    exact = float(lower.max())
    factor = min(1., exact * (1 + 1e-5) + 1e-7)
    margin = (D + factor * E) / (cc + bb * factor + aa * factor ** 2).sqrt()
    assert margin.min() >= target_margin - 1e-3, (head, factor, float(margin.min()))
    factors.append(factor)
    head_reports.append({'head': head, 'position': pos, 'original_minimum_margin': float(original_margin.min()),
                         'exact_lower_boundary': exact, 'selected_factor': factor,
                         'new_minimum_margin_before_rounding': float(margin.min())})
Qnew = (native + torch.tensor(factors, dtype=torch.float64)[:, None, None] * program).float()
newq = normal(torch.einsum('nc,hjc->nhj', SX[:, 8], Qnew.double())) * G
actual_margins = []
for head, pos in enumerate(positions):
    scores = (KR * newq[:, head, None]).sum(-1) / 16
    other = torch.tensor([j for j in range(9) if j != pos])
    actual_margins.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
assert min(actual_margins) >= target_margin - 1e-3
word_path = ROOT / 'artifacts/a52v1_value_word.pt'
word = torch.load(word_path, weights_only=True, mmap=True)
T = word['captures']['3.input'][384:].double().reshape(-1, 640)
hostq = normal(torch.einsum('nc,hjc->nhj', T, Qh)) * Gh
wordk = normal(T @ Wk.T) * Gk
CK = wordk.T @ wordk / len(wordk)
swap = torch.cat([torch.arange(128, 256), torch.arange(128)])
sign = torch.cat([-torch.ones(128), torch.ones(128)])
def rotate_cov(matrix, lag):
    c0 = (lag * freq).cos().repeat(2)
    s0 = (lag * freq).sin().repeat(2) * sign
    t = matrix * c0[:, None] + matrix[swap] * s0[:, None]
    return t * c0[None] + t[:, swap] * s0[None]
metric = sum(rotate_cov(CK, -lag) for lag in (1, 4, 16, 32, 64)) / (5 * 256)
def word_metric(matrix):
    error = normal(torch.einsum('nc,hjc->nhj', T, matrix)) * G - hostq
    return {'covariance_dot_error': float(((error @ metric) * error).sum(-1).mean()),
            'query_rms_change': float(error.square().mean().sqrt())}
report = {'routing_margin': target_margin, 'factor_rounding_guard': [1e-5, 1e-7],
          'scale_trials': [], 'format_examples': len(SX), 'heads': head_reports,
          'float32_margins': actual_margins, 'decomposition_maximum_error': decomp_error,
          'before': word_metric(Q), 'after': word_metric(Qnew.double()),
          'unchanged': 'Query normalization, K,V,O, all MLPs and other attention blocks',
          'labels': 'Source token positions only; no sums or answer tokens'}
print(json.dumps(report, indent=2), flush=True)
ch[qn] = Qnew.reshape(1024, 640)
c['exact_query_balance_a55'] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'query_balance.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', query_path, word_path,
             query_metadata_path, query_reference_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=bm['categories'])
