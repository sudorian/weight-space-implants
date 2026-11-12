"""Allocate A-query program norm gains by analytic diagonal constraints.

Each head's program-only score direction is rescaled to a declared routing
margin. A channel-weighted gain profile then satisfies every raw query norm
constraint via a finite exact breakpoint calculation. Shared ordinary-query
energy is repaired by one original-target linear solve per ridge. No sums,
gradients, iterative loss fitting, softmax, or model replay are involved.
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
assert 'query_gain_allocation_a61' not in circuit
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

program_direction = amplitude / amplitude.square().mean(1, keepdim=True).sqrt() * Gold
original_asymptotic = []
for head, pos in enumerate(positions):
    scores = (KR * program_direction[head]).sum(-1) / 16
    other = torch.tensor([j for j in range(9) if j != pos])
    original_asymptotic.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
original_asymptotic = torch.tensor(original_asymptotic, dtype=torch.float64)
assert original_asymptotic.min() > 8
raw_importance = (Hraw[:nt].square() / Henergy[:nt, :, None]).mean((0, 1))
ordinary_idx = ordinary.nonzero().flatten()
assert Gh.abs().min() > 1e-6
trials, best_state, best_cost = [], None, float('inf')
for asymptotic_margin in (8.25, 8.5, 9., 10., 12.):
    wanted = program_direction * (asymptotic_margin / original_asymptotic)[:, None]
    # This profile is the fourth-root allocation from a quadratic gain cost
    # under inverse-square raw-query norm constraints. The shared scalar is
    # solved exactly at the HOST-gain floor, not found by a numerical search.
    profile = (wanted.square().sum(0) / raw_importance.clamp_min(1e-12)).pow(.25)
    profile /= profile.max()
    profile[ordinary] = 0
    threshold = Gh[selected].square() / profile[selected].square().clamp_min(1e-30)
    sequence = threshold.argsort()
    idx = selected[sequence]
    sorted_threshold = threshold[sequence]
    head_levels = []
    for head in range(4):
        host_terms = wanted[head, idx].square() / Gh[idx].square()
        boost_terms = wanted[head, idx].square() / profile[idx].square()
        if host_terms.sum() <= 256:
            head_levels.append(0.)
            continue
        choices = []
        for k in range(1, len(idx) + 1):
            remaining = host_terms[k:].sum()
            if remaining >= 256:
                continue
            level = boost_terms[:k].sum() / (256 - remaining)
            low = sorted_threshold[k - 1]
            high = sorted_threshold[k] if k < len(idx) else torch.inf
            if level >= low * (1 - 1e-10) and level <= high * (1 + 1e-10):
                choices.append(float(level))
        assert choices
        head_levels.append(min(choices))
    level = max(head_levels) * (1 + 1e-8)
    gain0 = Gh.sign() * torch.maximum(Gh.abs(), level ** .5 * profile)
    program_raw_direction = wanted / gain0
    program_raw_direction /= program_raw_direction.square().mean(1, keepdim=True).sqrt()
    Pquery = program_raw_direction[:, :, None] * qnull
    Pr = torch.einsum('nc,hjc->nhj', SX[:, 8], Pquery)
    native = B * (Gh / gain0)[None, :, None]
    initial_factors, initial_routing = balance(native, gain0)
    assert initial_factors is not None, initial_routing
    initial_query = native + initial_factors[:, None, None] * Pquery
    Rraw = torch.einsum('nc,hjc->nhj', T, initial_query)
    Ereset = Rraw.square().mean(-1) + 1e-6
    target = 1 - Ereset / Henergy
    design = Rraw[:, :, ordinary_idx].square() / (256 * Henergy[:, :, None])
    X = design[:nt].reshape(-1, len(ordinary_idx))
    y = target[:nt].reshape(-1)
    standard = X.square().mean(0).sqrt().clamp_min(1e-30)
    Z = X / standard
    cov, rhs = Z.T @ Z / len(Z), Z.T @ y / len(Z)
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        R = cov.clone()
        R.diagonal().add_(ridge)
        beta = torch.linalg.solve(R, rhs) / standard
        row = {'asymptotic_margin': asymptotic_margin, 'ridge': ridge,
               'gain_profile_level_squared': level, 'head_level_constraints': head_levels,
               'selected_gains': gain0[selected].tolist(), 'eligible': False}
        if (1 + beta).min() <= 0:
            row.update(reason='Negative squared native row scale', minimum_scale_squared=float((1 + beta).min()))
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        scale = (1 + beta).sqrt()
        N = native.clone()
        N[:, ordinary_idx] *= scale[None, :, None]
        gain = gain0.clone()
        gain[ordinary_idx] /= scale
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
                   native_row_scales=scale.tolist())
        row['eligible'] = bool(min(actual_margins) >= 8 - 1e-3 and Qnew.abs().max() <= 1e6 and Gnew.abs().max() <= 1e6)
        trials.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('selected_gains','native_row_scales')}), flush=True)
        if row['eligible'] and row['covariance_dot_error'] < best_cost:
            best_cost = row['covariance_dot_error']
            best_state = Qnew.clone(), Gnew.clone(), row
assert best_state is not None
Qnew, Gnew, chosen = best_state
ch[qn], ch[gn] = Qnew.reshape(1024, 640), Gnew
report = {'asymptotic_margin_trials': [8.25, 8.5, 9., 10., 12.],
          'ridge_trials': [1e-6, 1e-4, .01, 1., 100.], 'routing_margin': 8,
          'gain_profile_power': .25, 'gain_floor': 'HOST absolute q_norm gain',
          'head_original_asymptotic_margins': original_asymptotic.tolist(),
          'training_tokens': nt, 'validation_tokens': len(T) - nt, 'format_examples': len(SX),
          'baseline': baseline, 'trials': trials, 'chosen': chosen,
          'changed': 'Only Q3 and its q_norm; native K,V,O and every MLP remain exact to the base',
          'targets': 'Source position routing and HOST query energy on unlabeled calibration text'}
circuit['query_gain_allocation_a61'] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', circuit)
atomic_json(dst.parent / 'query_gain_allocation.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', metadata_path,
             query_reference_path, query_path, word_path] + paths,
     arguments=vars(a), description=__doc__, extra=report, categories=bm['categories'])
