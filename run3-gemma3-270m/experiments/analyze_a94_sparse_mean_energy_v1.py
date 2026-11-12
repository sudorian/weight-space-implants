"""Sparse format-mean cancellation with a sparse raw-energy repair.

Every ordinary query row cancels the marker mean through physical coordinate
334. A subset of ordinary rows receives one original HOST raw-energy solve,
with inverse norm gains preserving native score numerators. Selection uses
recorded centered causal QK scores without attention or activation replay.
No gradients, answers or iterative weight fits occur.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a90v1_grouped_affine/delta.pt')
p.add_argument('--output', default='artifacts/a94_sparse_mean_energy_v1.pt')
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
prefix = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [prefix + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
Qh = h[qn].double().reshape(4, 256, 640)
Qold = ch[qn].double().reshape(4, 256, 640)
Gh, Gold = 1 + h[gn].double(), 1 + ch[gn].double()
Kh, Gk = h[kn].double(), 1 + h[kgn].double()
assert kn not in ch and kgn not in ch
mp = ROOT / 'experiments/a45v2_query_energy/circuit.pt'
metadata = torch.load(mp, weights_only=True)
selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
positions = metadata['native_key_query_layer3']['source_positions']
assert len(selected) == 64 and positions == [3, 2, 1, 0]
ordinary = torch.ones(256, dtype=torch.bool)
ordinary[selected] = False
paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
SX = torch.cat([torch.load(path, weights_only=True, mmap=True)['captures']['3.input'][:, :9]
                for path in paths]).double()
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
quiet = torch.load(qp, weights_only=True)['query'].double()
mean_input = SX[:1024, 8].mean(0)
qnull = quiet / (mean_input @ quiet)
mean_q = torch.einsum('c,hjc->hj', mean_input, Qh)
B = Qh - mean_q[:, :, None] * qnull
orth = B - (B @ quiet)[:, :, None] * quiet / quiet.square().sum()
coefficient = (Qold * orth).sum((0, 2)) / (B * orth).sum((0, 2))
residual = Qold - coefficient[None, :, None] * B
amplitude = (residual @ quiet) / quiet.square().sum()
amplitude[:, ordinary] = 0
Pquery = amplitude[:, :, None] * quiet
assert (residual - Pquery).abs().max() < 1e-6
wp = ROOT / 'artifacts/a52v1_value_word.pt'
T = torch.load(wp, weights_only=True, mmap=True)['captures']['3.input'].double().reshape(-1, 640)
nt = 384 * 64
normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
Hraw = torch.einsum('nc,hjc->nhj', T, Qh)
Henergy = Hraw.square().mean(-1) + 1e-6
Hnorm = normal(Hraw) * Gh
wordK = normal(T @ Kh.T) * Gk
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
swap = torch.cat([torch.arange(128, 256), torch.arange(128)])
sign = torch.cat([-torch.ones(128), torch.ones(128)])
def rotate_cov(matrix, lag):
    co = (lag * freq).cos().repeat(2)
    si = (lag * freq).sin().repeat(2) * sign
    tmp = matrix * co[:, None] + matrix[swap] * si[:, None]
    return tmp * co[None] + tmp[:, swap] * si[None]
def key_metric(keys):
    cov = keys.T @ keys / len(keys)
    return sum(rotate_cov(cov, -lag) for lag in (1, 4, 16, 32, 64)) / (5 * 256)
metric_train, metric_valid = key_metric(wordK[:nt]), key_metric(wordK[nt:])
importance = Hnorm[:nt].square().mean((0, 1)) * metric_train.diagonal()
importance[selected] = torch.inf
order = importance.argsort()
def rotate(z):
    length = z.shape[-2]
    angles = torch.arange(length, dtype=torch.float64)[:, None] * freq
    co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * co + torch.cat([-z[..., 128:], z[..., :128]], -1) * si
word_host_k = rotate(wordK[nt:].reshape(-1, 64, 256))
word_host_q = rotate(Hnorm[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
host_scores = word_host_q @ word_host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)
def word_metrics(matrix, gain):
    raw = torch.einsum('nc,hjc->nhj', T[nt:], matrix)
    norm = normal(raw) * gain
    error = norm - Hnorm[nt:]
    q = rotate(norm.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    difference = q @ word_host_k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (difference * causal).sum(-1) / counts
    variance = ((difference - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(variance.mean()),
            'paired_score_p99_query_rms': float(variance.sqrt().flatten().quantile(.99)),
            'covariance_dot_error': float(((error @ metric_valid) * error).sum(-1).mean()),
            'query_rms_change': float(error.square().mean().sqrt()),
            'relative_raw_energy_rms': float(((raw.square().mean(-1) + 1e-6) / Henergy[nt:] - 1).square().mean().sqrt())}
SK = normal(SX @ Kh.T) * Gk
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
KR = SK * co[None] + torch.cat([-SK[:, :, 128:], SK[:, :, :128]], -1) * si[None]
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
        Aq, Bq, Cq = E.square() - 64 * aa, 2 * D * E - 64 * bb, D.square() - 64 * cc
        if Aq.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Programme-only routing limit below8'}
        discriminant = Bq.square() - 4 * Aq * Cq
        upper = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * Aq), torch.zeros_like(Aq))
        bound = torch.maximum(upper, -D / E).clamp_min(0)
        factor = float(bound.max()) * (1 + 1e-5) + 1e-7
        value = (D + factor * E) / (cc + factor * bb + factor ** 2 * aa).sqrt()
        if value.min() < 8 - 1e-3:
            return None, {'reason': 'Analytic routing check failed', 'margin': float(value.min())}
        factors.append(factor)
        margins.append(float(value.min()))
    return torch.tensor(factors, dtype=torch.float64), {'factors': factors, 'minimum_margins': margins}
baseline = word_metrics(Qold, Gold)
print(json.dumps({'baseline': baseline}), flush=True)
trials, candidates = [], {}
qsp = torch.zeros(640, dtype=torch.float64)
qsp[334] = 1 / mean_input[334]
Nstart = Qh - mean_q[:, :, None] * qsp
Nstart[:, selected] = B[:, selected] * coefficient[None, selected, None]
gain0 = Gh.clone()
gain0[selected] = Gold[selected]
factors0, routing0 = balance(Nstart, gain0)
assert factors0 is not None, routing0
initial = Nstart + factors0[:, None, None] * Pquery
Rraw = torch.einsum('nc,hjc->nhj', T[:nt], initial)
Ereset = Rraw.square().mean(-1) + 1e-6
y = (1 - Ereset / Henergy[:nt]).reshape(-1)
ordinary_idx = ordinary.nonzero().flatten()
all_design = (Rraw[:, :, ordinary_idx].square() / (256 * Henergy[:nt, :, None])).reshape(-1, len(ordinary_idx))
centered = all_design - all_design.mean(0)
correlation = (centered * (y - y.mean())[:, None]).mean(0).abs() / centered.square().mean(0).sqrt().clamp_min(1e-30)
order = ordinary_idx[correlation.argsort(descending=True)]
del centered, Rraw
for count in (32, 64, 96, 128, 160):
    retained = order[:count]
    native = Nstart.clone()
    mapping = torch.tensor([(ordinary_idx == i).nonzero().item() for i in retained])
    X = all_design[:, mapping]
    standard = X.square().mean(0).sqrt().clamp_min(1e-30)
    Z = X / standard
    cov, rhs = Z.T @ Z / len(Z), Z.T @ y / len(Z)
    del X, Z
    best = None
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        row = {'ordinary_rows_retained': count, 'ridge': ridge, 'eligible': False}
        N, gain = native.clone(), gain0.clone()
        if count:
            R = cov.clone()
            R.diagonal().add_(ridge)
            beta = torch.linalg.solve(R, rhs) / standard
            if (1 + beta).min() <= 0:
                row.update(reason='Negative squared row scale', minimum_scale_squared=float((1 + beta).min()))
                trials.append(row)
                print(json.dumps(row), flush=True)
                continue
            scale = (1 + beta).sqrt()
            N[:, retained] *= scale[None, :, None]
            gain[retained] /= scale
        factors, routing = balance(N, gain)
        row.update(routing)
        if factors is None:
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        Qnew = (N + factors[:, None, None] * Pquery).float()
        Gnew = (gain - 1).float()
        restored = ordinary.clone()
        restored[retained] = False
        keep = restored[:, None].expand(256, 640).clone()
        keep[:, 334] = False
        assert torch.equal(Qnew[:, keep], h[qn].reshape(4, 256, 640)[:, keep])
        Gnew[restored] = h[gn][restored]
        actual_gain = 1 + Gnew.double()
        actual_q = normal(torch.einsum('nc,hjc->nhj', SX[:, 8], Qnew.double())) * actual_gain
        margins = []
        for head, pos in enumerate(positions):
            scores = (KR * actual_q[:, head, None]).sum(-1) / 16
            other = torch.tensor([i for i in range(9) if i != pos])
            margins.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
        row.update(float32_margins=margins, **word_metrics(Qnew.double(), actual_gain),
                   changed_query_scalars=int((Qnew.view(torch.int32).reshape(1024, 640) != h[qn].view(torch.int32)).sum()),
                   maximum_query_weight=float(Qnew.abs().max()), maximum_norm_weight=float(Gnew.abs().max()))
        row['eligible'] = bool(min(margins) >= 8 - 1e-3 and Qnew.abs().max() <= 1e6 and Gnew.abs().max() <= 1e6)
        trials.append(row)
        print(json.dumps(row), flush=True)
        cost = row['paired_centered_causal_score_mse']
        if row['eligible'] and (best is None or cost < best[0]):
            best = cost, Qnew.clone(), Gnew.clone(), row
    if best is not None:
        candidates[str(count)] = {'query': best[1], 'gain_parameter': best[2], 'report': best[3], 'retained': retained}
best_key = min(candidates, key=lambda k: candidates[k]['report']['paired_centered_causal_score_mse']) if candidates else None
report = {'retained_ordinary_row_trials': [32, 64, 96, 128, 160], 'ridge_trials': [1e-6, 1e-4, .01, 1, 100],
          'routing_margin': 8, 'selection': 'largest univariate centered correlation with the original HOST energy deficit',
          'native_mean_projection_coordinate': 334, 'initial_routing': routing0,
          'training_tokens': nt, 'validation_tokens': len(T) - nt, 'format_examples': len(SX),
          'baseline': baseline, 'trials': trials, 'best_key': best_key,
          'source_sha256': sha(Path(__file__)), 'inputs': {str(x.relative_to(ROOT)): sha(x) for x in [base, mp, qp, wp] + paths},
          'no_answer_targets_activation_replay_or_iterative_fit': True}
out = ROOT / a.output
assert not out.exists()
atomic_torch(out, {'report': report, 'candidates': candidates})
atomic_json(out.with_suffix('.json'), report)
print(json.dumps({'best_key': best_key, 'best': candidates[best_key]['report'] if best_key else None}), flush=True)
