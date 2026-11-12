"""Restore native A-query energy using zero-output normalization carriers.

A few quiet query channels have q_norm gain zero and carry a low-rank factor
of the missing native raw-query energy. All other native query numerators
are restored algebraically. Carrier directions come from calibration PCA,
and their squared-energy scales from independent closed-form linear solves.
Program query factors satisfy analytic quadratic source-routing constraints.
No gradients, iterative weight fitting, answers, softmax or model replay.
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
circuit = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'query_energy_carriers_a60' not in circuit
metadata_path = ROOT / 'experiments/a45v2_query_energy/circuit.pt'
metadata = torch.load(metadata_path, weights_only=True)
query_reference_path = metadata_path.parent / 'delta.pt'
reference = expand(h, query_reference_path)
prefix = 'model.layers.3.self_attn.'
qn, gn = prefix + 'q_proj.weight', prefix + 'q_norm.weight'
assert torch.equal(ch[qn], reference[qn]) and torch.equal(ch[gn], reference[gn])
del reference
Qh = h[qn].double().reshape(4, 256, 640)
Qold = ch[qn].double().reshape(4, 256, 640)
Gh, Gold = 1 + h[gn].double(), 1 + ch[gn].double()
Kweight = h[prefix + 'k_proj.weight'].double()
Kgain = 1 + h[prefix + 'k_norm.weight'].double()
paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
SX = torch.cat([torch.load(path, weights_only=True, mmap=True)['captures']['3.input'][:, :9]
                for path in paths]).double()
mean_input = SX[:1024, 8].mean(0)
query_path = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
qfull = torch.load(query_path, weights_only=True)['query'].double()
qnull = qfull / (mean_input @ qfull)
mean_q = torch.einsum('c,hjc->hj', mean_input, Qh)
B = Qh - mean_q[:, :, None] * qnull
orth = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
coefficient = (Qold * orth).sum((0, 2)) / (B * orth).sum((0, 2))
residual = Qold - coefficient[None, :, None] * B
amplitude = (residual @ qfull) / qfull.square().sum()
selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
ordinary = torch.ones(256, dtype=torch.bool)
ordinary[selected] = False
amplitude[:, ordinary] = 0
Pquery = amplitude[:, :, None] * qfull
assert (residual - Pquery).abs().max() < 1e-6
Gbase = Gh.clone()
Gbase[selected] = Gold[selected]
scales = torch.ones(256, dtype=torch.float64)
scales[selected] = Gh[selected] / Gold[selected]
Nbase = B * scales[None, :, None]
word_path = ROOT / 'artifacts/a52v1_value_word.pt'
T = torch.load(word_path, weights_only=True, mmap=True)['captures']['3.input'].double().reshape(-1, 640)
nt = 384 * 64
normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
Hraw = torch.einsum('nc,hjc->nhj', T, Qh)
Henergy = Hraw.square().mean(-1) + 1e-6
Hnorm = normal(Hraw) * Gh
wordK = normal(T @ Kweight.T) * Kgain
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
swap = torch.cat([torch.arange(128, 256), torch.arange(128)])
sign = torch.cat([-torch.ones(128), torch.ones(128)])
def rotate_cov(matrix, lag):
    co = (lag * freq).cos().repeat(2)
    si = (lag * freq).sin().repeat(2) * sign
    t = matrix * co[:, None] + matrix[swap] * si[:, None]
    return t * co[None] + t[:, swap] * si[None]
def key_metric(keys):
    cov = keys.T @ keys / len(keys)
    return sum(rotate_cov(cov, -lag) for lag in (1, 4, 16, 32, 64)) / (5 * 256)
metric_train, metric_valid = key_metric(wordK[:nt]), key_metric(wordK[nt:])
importance = Hnorm[:nt].square().mean((0, 1)) * metric_train.diagonal()
importance[selected] = torch.inf
order = importance.argsort()
Cinput = T[:nt].T @ T[:nt] / nt
ev, evec = torch.linalg.eigh(Cinput)
ev = ev.clamp_min(float(ev.max()) * 1e-12)
cov_root = evec * ev.sqrt()[None]
cov_inverse_root = evec.T / ev.sqrt()[:, None]
SK = normal(SX @ Kweight.T) * Kgain
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
KR = SK * co[None] + torch.cat([-SK[:, :, 128:], SK[:, :, :128]], -1) * si[None]
positions = metadata['native_key_query_layer3']['source_positions']
assert positions == [3, 2, 1, 0]
Pr = torch.einsum('nc,hjc->nhj', SX[:, 8], Pquery)
def balance(native, gain):
    Nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
    factors, margins = [], []
    for head, pos in enumerate(positions):
        other = torch.tensor([i for i in range(9) if i != pos])
        kd = KR[:, pos, None] - KR[:, other]
        D = (kd * (Nr[:, head] * gain)[:, None]).sum(-1) / 16
        E = (kd * (Pr[:, head] * gain)[:, None]).sum(-1) / 16
        aa = Pr[:, head].square().mean(1, keepdim=True)
        bb = 2 * (Nr[:, head] * Pr[:, head]).mean(1, keepdim=True)
        cc = Nr[:, head].square().mean(1, keepdim=True) + 1e-6
        A, Bq, Cq = E.square() - 64 * aa, 2 * D * E - 64 * bb, D.square() - 64 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'The program-only asymptotic margin is below8',
                          'minimum_asymptotic_margin': float((E / aa.sqrt()).min())}
        discriminant = Bq.square() - 4 * A * Cq
        upper = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A),
                            torch.zeros_like(A))
        bound = torch.maximum(upper, -D / E).clamp_min(0)
        factor = float(bound.max()) * (1 + 1e-5) + 1e-7
        value = (D + factor * E) / (cc + factor * bb + factor ** 2 * aa).sqrt()
        if value.min() < 8 - 1e-3:
            return None, {'reason': 'Analytic routing check failed', 'margin': float(value.min())}
        factors.append(factor)
        margins.append(float(value.min()))
    return torch.tensor(factors, dtype=torch.float64), {'factors': factors, 'minimum_margins': margins}
def word_metrics(matrix, gain):
    raw = torch.einsum('nc,hjc->nhj', T[nt:], matrix)
    error = normal(raw) * gain - Hnorm[nt:]
    return {'covariance_dot_error': float(((error @ metric_valid) * error).sum(-1).mean()),
            'query_rms_change': float(error.square().mean().sqrt()),
            'relative_raw_energy_rms': float(((raw.square().mean(-1) + 1e-6) / Henergy[nt:] - 1).square().mean().sqrt())}
baseline = word_metrics(Qold, Gold)
print(json.dumps({'baseline': baseline}), flush=True)
trials, best_state, best_cost = [], None, float('inf')
for count in (1, 2, 4, 8, 16):
    carriers = order[:count]
    native = Nbase.clone()
    native[:, carriers] = 0
    gain = Gbase.clone()
    gain[carriers] = 0
    directions, eig_reports = [], []
    for head in range(4):
        missing = Qh[head].T @ Qh[head] - native[head].T @ native[head]
        whitened = cov_root.T @ missing @ cov_root
        vals, vecs = torch.linalg.eigh((whitened + whitened.T) / 2)
        vals, vecs = vals.flip(0), vecs.flip(1)
        assert vals[count - 1] > 0
        rows = vals[:count].sqrt()[:, None] * (vecs[:, :count].T @ cov_inverse_root)
        rows -= (rows @ mean_input)[:, None] * qnull
        directions.append(rows)
        eig_reports.append({'head': head, 'positive_energy_capture_fraction': float(vals[:count].sum() / vals.clamp_min(0).sum()),
                            'negative_energy_fraction': float((-vals.clamp_max(0)).sum() / vals.clamp_min(0).sum())})
    directions = torch.stack(directions)
    carrier_raw = torch.einsum('nc,hjc->nhj', T, directions)
    base_raw = torch.einsum('nc,hjc->nhj', T, native + Pquery)
    base_energy = base_raw.square().mean(-1) + 1e-6
    target = 1 - base_energy / Henergy
    design = carrier_raw.square() / (256 * Henergy[:, :, None])
    standard = design[:nt].square().mean(0).sqrt().clamp_min(1e-30)
    Z = design[:nt] / standard
    cov = torch.einsum('nhi,nhj->hij', Z, Z) / nt
    rhs = torch.einsum('nhi,nh->hi', Z, target[:nt]) / nt
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        R = cov.clone()
        R.diagonal(dim1=-2, dim2=-1).add_(ridge)
        beta = torch.linalg.solve(R, rhs[:, :, None])[:, :, 0] / standard
        row = {'carrier_count': count, 'carriers': carriers.tolist(), 'ridge': ridge,
               'energy_eigenspaces': eig_reports, 'eligible': False}
        if beta.min() <= 0:
            row.update(reason='Negative squared carrier scale', minimum_scale_squared=float(beta.min()))
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        N = native.clone()
        N[:, carriers] = directions * beta.sqrt()[:, :, None]
        factors, routing = balance(N, gain)
        row.update(routing)
        if factors is None:
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        Qnew = (N + factors[:, None, None] * Pquery).float()
        Gnew = (gain - 1).float()
        actual_gain = 1 + Gnew.double()
        actual_q = normal(torch.einsum('nc,hjc->nhj', SX[:, 8], Qnew.double())) * actual_gain
        actual_margins = []
        for head, pos in enumerate(positions):
            scores = (KR * actual_q[:, head, None]).sum(-1) / 16
            other = torch.tensor([i for i in range(9) if i != pos])
            actual_margins.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
        row.update(float32_margins=actual_margins, **word_metrics(Qnew.double(), actual_gain),
                   maximum_query_weight=float(Qnew.abs().max()), maximum_norm_weight=float(Gnew.abs().max()),
                   energy_scale_squared=beta.tolist())
        row['eligible'] = min(actual_margins) >= 8 - 1e-3 and Qnew.abs().max() <= 1e6 and Gnew.abs().max() <= 1e6
        row['eligible'] = bool(row['eligible'])
        trials.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and row['covariance_dot_error'] < best_cost:
            best_cost = row['covariance_dot_error']
            best_state = Qnew.clone(), Gnew.clone(), row
assert best_state is not None
Qnew, Gnew, chosen = best_state
ch[qn], ch[gn] = Qnew.reshape(1024, 640), Gnew
report = {'carrier_count_trials': [1, 2, 4, 8, 16], 'ridge_trials': [1e-6, 1e-4, .01, 1., 100.],
          'routing_margin': 8, 'factor_guard': [1e-5, 1e-7], 'input_covariance_eigenvalue_floor_ratio': 1e-12,
          'training_tokens': nt, 'validation_tokens': len(T) - nt, 'format_examples': len(SX),
          'baseline': baseline, 'trials': trials, 'chosen': chosen,
          'carrier_selection': 'Least HOST native query diagonal score energy outside program support',
          'changed': 'Only Q3 and its q_norm; native K,V,O and every MLP remain exact to the base',
          'target': 'Missing HOST raw-query squared energy on unlabeled calibration text'}
circuit['query_energy_carriers_a60'] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', circuit)
atomic_json(dst.parent / 'energy_carriers.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', metadata_path,
             query_reference_path, query_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=bm['categories'])
