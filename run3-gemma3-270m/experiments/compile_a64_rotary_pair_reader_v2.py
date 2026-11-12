"""Read A through one programmed rotary pair and unchanged native V/O.

A linear key function predicts native key RMS on recorded format positions;
these scalar normalization targets contain no digit values or answers. The
query phase names the fixed source position. Other K rows are HOST; ordinary Q rows cancel the format mean using the
existing calibration-quiet format readout. This version2 removes the large
format-native query offset before setting the program gain.
Native norm gains are one-shot calibration numerator accommodations, and the
program gain is the exact linear bound for routing margin8. No attention
softmax, GELU, gradients, iterative weight fitting, or model replay.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a61v1_query_gain_allocation/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
circuit = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'rotary_pair_reader_a64' not in circuit
prefix = 'model.layers.3.self_attn.'
qn, kn, qgn, kgn = [prefix + name for name in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight')]
Qh, Kh = h[qn].double().reshape(4, 256, 640), h[kn].double()
Gh, Hk = 1 + h[qgn].double(), 1 + h[kgn].double()
paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
SX = torch.cat([torch.load(path, weights_only=True, mmap=True)['captures']['3.input'][:, :9]
                for path in paths]).double()
query_path = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
qrow = torch.load(query_path, weights_only=True)['query'].double()
qrow /= SX[:1024, 8].mean(0) @ qrow
assert (SX[:, 8] @ qrow).min() > 0
word_path = ROOT / 'artifacts/a52v1_value_word.pt'
T = torch.load(word_path, weights_only=True, mmap=True)['captures']['3.input'].double().reshape(-1, 640)
nt = 384 * 64
positions = [3, 2, 1, 0]
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
normal = lambda z: z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def rotate(z):
    length = z.shape[-2]
    angles = torch.arange(length, dtype=torch.float64)[:, None] * freq
    co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * co + torch.cat([-z[..., 128:], z[..., :128]], -1) * si
hostQ = torch.einsum('nc,hjc->nhj', T, Qh)
hostK = T @ Kh.T
HQ, HK = normal(hostQ) * Gh, normal(hostK) * Hk
word_host_q = rotate(HQ[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
word_host_k = rotate(HK[nt:].reshape(-1, 64, 256))
host_scores = word_host_q @ word_host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)
def paired_cost(Q, K, gq, gk):
    q = normal(torch.einsum('nc,hjc->nhj', T[nt:], Q)) * gq
    k = normal(T[nt:] @ K.T) * gk
    q = rotate(q.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    k = rotate(k.reshape(-1, 64, 256))
    difference = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (difference * causal).sum(-1) / counts
    variance = ((difference - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(variance.mean()),
            'paired_score_p99_query_rms': float(variance.sqrt().flatten().quantile(.99)),
            'paired_score_max_query_rms': float(variance.sqrt().max())}
baseline = paired_cost(ch[qn].double().reshape(4, 256, 640), ch.get(kn, h[kn]).double(),
                       1 + ch[qgn].double(), 1 + ch.get(kgn, h[kgn]).double())
print(json.dumps({'baseline': baseline}), flush=True)
key_scale = .001
key_reports, trials, best_state, best_cost = [], [], None, float('inf')
for pair in (4, 5, 6, 7, 8):
    channels = torch.tensor([pair, pair + 128])
    K0 = Kh.clone()
    K0[channels] = 0
    native_key_raw = SX @ K0.T
    key_rms = (native_key_raw.square().mean(-1) + 1e-6).sqrt()
    X = (SX[:1024] / key_rms[:1024, :, None]).reshape(-1, 640)
    std = X.square().mean(0).sqrt().clamp_min(1e-12)
    Z = X / std
    cov, rhs = Z.T @ Z / len(Z), Z.mean(0)
    key_choices = []
    for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
        R = cov.clone()
        R.diagonal().add_(ridge)
        key_function = torch.linalg.solve(R, rhs) / std
        K = K0.clone()
        K[pair] = key_scale * key_function
        K = K.float().double()
        nk = normal(SX @ K.T)
        gaps = []
        for head, pos in enumerate(positions):
            relative_angle = (torch.arange(9, dtype=torch.float64) - pos) * freq[pair]
            values = nk[:, :, pair] * relative_angle.cos()[None]
            others = torch.tensor([i for i in range(9) if i != pos])
            gaps.append(values[:, pos, None] - values[:, others])
        minimum = float(torch.stack(gaps, 1).min())
        ratio = (SX @ key_function) / key_rms
        row = {'pair': pair, 'ridge': ridge, 'minimum_key_phase_gap': minimum,
               'key_ratio_minimum': float(ratio.min()), 'key_ratio_maximum': float(ratio.max()),
               'fresh_key_ratio_rms_error': float((ratio[1024:] - 1).square().mean().sqrt())}
        key_reports.append(row)
        print(json.dumps({'key_geometry': row}), flush=True)
        if minimum > 0:
            key_choices.append((minimum, K, key_function, row))
    if not key_choices:
        continue
    _, K, key_function, key_choice = max(key_choices, key=lambda v: v[0])
    rawk = T @ K.T
    nk_word = normal(rawk)
    gk = Hk * (normal(hostK)[:nt] * nk_word[:nt]).sum(0) / nk_word[:nt].square().sum(0).clamp_min(1e-30)
    gk[channels] = 0
    nk = normal(SX @ K.T)
    phase = (torch.tensor(positions, dtype=torch.float64) - 8) * freq[pair]
    for query_scale in (.01, .03, .1, .3, 1.):
        mean_q = torch.einsum('c,hjc->hj', SX[:1024, 8].mean(0), Qh)
        Q = Qh - mean_q[:, :, None] * qrow
        Q[:, pair] = query_scale * phase.cos()[:, None] * qrow
        Q[:, pair + 128] = query_scale * phase.sin()[:, None] * qrow
        Q = Q.float().double()
        rawq = torch.einsum('nc,hjc->nhj', T, Q)
        nq_word = normal(rawq)
        gq = Gh * (normal(hostQ)[:nt] * nq_word[:nt]).sum((0, 1)) / nq_word[:nt].square().sum((0, 1)).clamp_min(1e-30)
        gq[channels] = 0
        nq = normal(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
        native_q, native_k = rotate(nq * gq), rotate(nk * gk)
        native_scores = (native_q[:, :, 8, None] * native_k[:, None]).sum(-1) / 16
        program_q, program_k = torch.zeros_like(nq), torch.zeros_like(nk)
        program_q[..., channels], program_k[..., channels] = nq[..., channels], nk[..., channels]
        program_scores = (rotate(program_q)[:, :, 8, None] * rotate(program_k)[:, None]).sum(-1) / 16
        needed, min_gaps = [], []
        for head, pos in enumerate(positions):
            other = torch.tensor([i for i in range(9) if i != pos])
            gap = program_scores[:, head, pos, None] - program_scores[:, head, other]
            assert gap.min() > 0
            needed.append(float(((8 + native_scores[:, head, other] - native_scores[:, head, pos, None]) / gap).max()))
            min_gaps.append(float(gap.min()))
        gain = math.sqrt(max(needed) * (1 + 1e-5))
        if gain > 1e6:
            continue
        gq_new, gk_new = gq.clone(), gk.clone()
        gq_new[channels], gk_new[channels] = gain, gain
        qgain, kgain = (gq_new - 1).float(), (gk_new - 1).float()
        actual_q, actual_k = rotate(nq * (1 + qgain.double())), rotate(nk * (1 + kgain.double()))
        scores = (actual_q[:, :, 8, None] * actual_k[:, None]).sum(-1) / 16
        margins = []
        for head, pos in enumerate(positions):
            other = torch.tensor([i for i in range(9) if i != pos])
            margins.append(float((scores[:, head, pos] - scores[:, head, other].max(1).values).min()))
        assert min(margins) > 7.999
        row = {'pair': pair, 'query_scale': query_scale, 'key_scale': key_scale,
               'program_gain': gain, 'key_geometry': key_choice, 'minimum_program_gaps': min_gaps,
               'float32_source_margins': margins,
               **paired_cost(Q, K, 1 + qgain.double(), 1 + kgain.double())}
        trials.append(row)
        print(json.dumps(row), flush=True)
        cost = row['paired_centered_causal_score_mse']
        if cost < best_cost:
            best_cost = cost
            best_state = Q.float(), K.float(), qgain, kgain, key_function, row
report = {'baseline': baseline, 'pair_trials': [4, 5, 6, 7, 8],
          'key_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'query_scale_trials': [.01, .03, .1, .3, 1.], 'key_scale': key_scale,
          'routing_margin': 8, 'gain_square_rounding_guard': 1e-5,
          'training_tokens': nt, 'validation_tokens': len(T) - nt, 'format_examples': len(SX),
          'key_geometry_trials': key_reports, 'trials': trials,
          'target': 'Scalar native key RMS, source positions and native calibration query/key numerators',
          'native_value_output_and_all_MLP_weights_unchanged': True}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
if best_state is None:
    report['failure'] = 'No positive source-key phase gap among the declared constructions'
    atomic_json(dst.parent / 'rotary_pair_reader.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0)
Q, K, qgain, kgain, key_function, chosen = best_state
ch[qn], ch[kn], ch[qgn], ch[kgn] = Q.reshape(1024, 640), K, qgain, kgain
report['chosen'] = chosen
circuit['rotary_pair_reader_a64'] = report
atomic_torch(dst.parent / 'circuit.pt', circuit)
atomic_torch(dst.parent / 'reader_geometry.pt', {'key_function': key_function, 'query': qrow, 'report': report})
atomic_json(dst.parent / 'rotary_pair_reader.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (qn, kn, qgn, kgn):
    categories[name] = {'circuit_and_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', query_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
