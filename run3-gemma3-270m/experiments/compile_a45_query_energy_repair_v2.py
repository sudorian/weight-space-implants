"""Restore dense-query normalization energy using up to all ordinary channels.

The selected program rows retain their exact additive format-query component.
First restore the native query numerator after the earlier average-gain
accommodation. Then fit missing squared raw-query energy on unlabeled text
with one linear solve per declared anchor count and ridge. Reciprocal norm
gains preserve each anchor's native numerator. No model execution, answer
targets, gradients, or iterative fitting occurs here.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--base', default='experiments/a25v4_native_keys2/delta.pt')
p.add_argument('--layer', type=int, choices=[2, 3], default=2)
p.add_argument('--capture-prefix')
p.add_argument('--query')
a = p.parse_args()
layer = a.layer
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
circuit = torch.load(base.parent / 'circuit.pt', weights_only=True)
prior = circuit[f'native_key_query_layer{layer}']
selected = torch.tensor(prior['selected_channels'])
ordinary = torch.tensor([j for j in range(256) if j not in selected.tolist()])
if layer == 2:
    names = ['a1_detail_syntax_a', 'a1_detail_syntax_b', 'a1_syntax_stress_key']
    word_name = 'a1_detail_calib256'
    readerpath = ROOT / 'experiments/a2v7_reader/circuit.pt'
else:
    assert a.capture_prefix and a.query
    names = [f'{a.capture_prefix}_syntax_a', f'{a.capture_prefix}_syntax_stress']
    word_name = f'{a.capture_prefix}_calib256'
    readerpath = ROOT / 'experiments/a4v2_reader/circuit.pt'
querypath = ROOT / (a.query or f'artifacts/a25_quiet_query_layer{layer}_v1.pt')
caps = [torch.load(ROOT / f'artifacts/{n}.pt', weights_only=True, mmap=True) for n in names]
word = torch.load(ROOT / f'artifacts/{word_name}.pt', weights_only=True, mmap=True)
SX = torch.cat([c['captures'][f'{layer}.input'][:, :9] for c in caps]).double()
T = word['captures'][f'{layer}.input'].double().reshape(-1, 640)
nt = 192 * 64
qfull = torch.load(querypath, weights_only=True)['query'].double()
qnull = qfull / (SX[:, 8] @ qfull).mean()
pre = f'model.layers.{layer}.self_attn.'
Qhost = h[pre + 'q_proj.weight'].double().reshape(4, 256, 640)
Qbase = Qhost - torch.einsum('c,hjc->hj', SX[:, 8].mean(0), Qhost)[:, :, None] * qnull
Qcurrent = ch[pre + 'q_proj.weight'].double().reshape(4, 256, 640)
Ghost = 1 + h[pre + 'q_norm.weight'].double()
Gcurrent = 1 + ch[pre + 'q_norm.weight'].double()
assert torch.equal(ch.get(pre + 'k_proj.weight', h[pre + 'k_proj.weight']), h[pre + 'k_proj.weight'])
assert torch.equal(ch.get(pre + 'k_norm.weight', h[pre + 'k_norm.weight']), h[pre + 'k_norm.weight'])

# Recover the old shared native coefficient without fitting any labels.
# Orthogonal projection removes the known rank-one format-query addend.
B = Qbase[:, selected]
orth = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
coefficient = (Qcurrent[:, selected] * orth).sum((0, 2)) / (B * orth).sum((0, 2))
Qreset = Qcurrent.clone()
Qreset[:, selected] += (Ghost[selected] / Gcurrent[selected] - coefficient)[None, :, None] * B
Greset = Ghost.clone()
Greset[selected] = Gcurrent[selected]

normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
Hraw = torch.einsum('nc,hjc->nhj', T, Qhost)
Hnorm = normal(Hraw) * Ghost
R = torch.einsum('nc,hjc->nhj', T, Qreset)
N = R * Greset
Ehost = Hraw.square().mean(-1) + 1e-6
Ereset = R.square().mean(-1) + 1e-6
target = 1 - Ereset / Ehost

Wk = h[pre + 'k_proj.weight'].double()
Gk = 1 + h[pre + 'k_norm.weight'].double()
K = normal(T @ Wk.T) * Gk
CK = K[nt:].T @ K[nt:] / len(K[nt:])
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
swap = torch.cat([torch.arange(128, 256), torch.arange(128)])
sgn = torch.cat([-torch.ones(128), torch.ones(128)])

def rotate_cov(M, lag):
    c = (lag * freq).cos().repeat(2)
    s = (lag * freq).sin().repeat(2) * sgn
    A = M * c[:, None] + M[swap] * s[:, None]
    return A * c[None] + A[:, swap] * s[None]

metric = sum(rotate_cov(CK, -lag) for lag in (1, 4, 16, 32, 64)) / (5 * 256)
NV = N[nt:]
HV = Hnorm[nt:]
AA = ((NV @ metric) * NV).sum(-1)
BB = ((NV @ metric) * HV).sum(-1)
CC = ((HV @ metric) * HV).sum(-1)

def covariance_cost(energy):
    e = energy[nt:]
    return float((AA / e - 2 * BB / e.sqrt() + CC).mean().clamp_min(0))

Sraw = torch.einsum('nc,hjc->nhj', SX[:, 8], Qreset)
Senergy = Sraw.square().mean(-1) + 1e-6
SK = normal(SX @ Wk.T) * Gk
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
cs = angles.cos().repeat(1, 2)
sn = angles.sin().repeat(1, 2)
KR = SK * cs[None] + torch.cat([-SK[:, :, 128:], SK[:, :, :128]], -1) * sn[None]
positions = prior['source_positions']
source_gap = []
for head, pos in enumerate(positions):
    scores = (KR * (Sraw[:, head] * Greset)[:, None]).sum(-1) / 16
    other = torch.tensor([j for j in range(9) if j != pos])
    source_gap.append(scores[:, pos] - scores[:, other].max(1).values)
source_gap = torch.stack(source_gap, 1)

# Rank shared channels by ordinary native raw energy, using training text only.
importance = (R[:nt].square() / Ehost[:nt, :, None]).mean((0, 1))
importance[selected] = -torch.inf
order = importance.argsort(descending=True)
trials, choices = [], []
for count in (32, 64, 96, 128, 192):
    anchors = order[:count]
    design = R[:, :, anchors].square() / (256 * Ehost[:, :, None])
    X = design[:nt].reshape(-1, count)
    y = target[:nt].reshape(-1)
    rms = X.square().mean(0).sqrt().clamp_min(1e-30)
    Z = X / rms
    cov = Z.T @ Z / len(Z)
    rhs = Z.T @ y / len(Z)
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        C = cov.clone()
        C.diagonal().add_(ridge)
        beta = torch.linalg.solve(C, rhs) / rms
        rec = {'anchor_count': count, 'anchors': anchors.tolist(), 'ridge': ridge,
               'energy_coefficients': beta.tolist()}
        if (1 + beta).min() <= 0:
            rec['eligible'] = False
            rec['reason'] = 'negative squared row scale'
            trials.append(rec)
            choices.append(None)
            print(json.dumps(rec), flush=True)
            continue
        scales = (1 + beta).sqrt()
        energy = Ereset + (R[:, :, anchors].square() * beta).sum(-1) / 256
        se = Senergy + (Sraw[:, :, anchors].square() * beta).sum(-1) / 256
        margins = (source_gap / se[:, :len(positions)].sqrt()).min(0).values
        Qnew = Qreset.clone()
        Qnew[:, anchors] *= scales[None, :, None]
        Gnew = Greset.clone()
        Gnew[anchors] /= scales
        rec.update(row_scales=scales.tolist(), minimum_routing_margins=margins.tolist(),
                   covariance_dot_error=covariance_cost(energy),
                   validation_relative_energy_rms=float((energy[nt:] / Ehost[nt:] - 1).square().mean().sqrt()),
                   normalized_query_word_rms_change=float((N[nt:] / energy[nt:, :, None].sqrt() - HV).square().mean().sqrt()),
                   max_query_weight=float(Qnew.abs().max()), max_norm_weight=float((Gnew - 1).abs().max()))
        rec['eligible'] = bool(margins.min() >= 8 - 1e-3 and Qnew.abs().max() <= 1e6 and (Gnew - 1).abs().max() <= 1e6)
        trials.append(rec)
        choices.append((Qnew.float(), (Gnew - 1).float()))
        print(json.dumps({k: v for k, v in rec.items() if k not in ('anchors', 'energy_coefficients', 'row_scales')}), flush=True)
viable = [i for i, r in enumerate(trials) if r['eligible']]
assert viable, 'No shared energy repair preserves all source-routing margins'
best = min(viable, key=lambda i: trials[i]['covariance_dot_error'])
Qnew, gnew = choices[best]
actual_raw = torch.einsum('nc,hjc->nhj', SX[:, 8], Qnew.double())
actual_q = normal(actual_raw) * (1 + gnew.double())
actual_margins = []
for head, pos in enumerate(positions):
    scores = (KR * actual_q[:, head, None]).sum(-1) / 16
    other = torch.tensor([j for j in range(9) if j != pos])
    actual_margins.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
assert min(actual_margins) >= 8 - 1e-3
oldraw = torch.einsum('nc,hjc->nhj', T[nt:], Qcurrent)
old_error = normal(oldraw) * Gcurrent - HV
old_cost = float((((old_error @ metric) * old_error).sum(-1)).mean())
report = {'layer': layer, 'anchor_count_trials': [32, 64, 96, 128, 192],
          'ridge_trials': [1e-6, 1e-4, .01, 1., 100.], 'training_tokens': nt,
          'validation_tokens': len(T) - nt, 'selected_program_channels': selected.tolist(),
          'old_native_coefficients': coefficient.tolist(),
          'baseline_covariance_dot_error': old_cost,
          'baseline_normalized_query_word_rms_change': float(old_error.square().mean().sqrt()),
          'reset_covariance_dot_error': covariance_cost(Ereset),
          'trials': trials, 'chosen': trials[best],
          'float32_minimum_routing_margins': actual_margins,
          'target': 'missing squared native raw-query energy; unlabeled calibration text only',
          'format_labels': 'current source token positions only', 'metric_lags': [1, 4, 16, 32, 64]}
ch[pre + 'q_proj.weight'] = Qnew.reshape(1024, 640)
ch[pre + 'q_norm.weight'] = gnew
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
circuit[f'query_energy_repair_layer{layer}'] = report
atomic_torch(dst.parent / 'circuit.pt', circuit)
atomic_json(dst.parent / 'energy_repair.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
anchors = torch.tensor(trials[best]['anchors'])
for name in (pre + 'q_proj.weight', pre + 'q_norm.weight'):
    actual = ch[name].view(torch.int32) != h[name].view(torch.int32)
    mask = torch.zeros_like(actual)
    if actual.ndim == 2:
        mask.reshape(4, 256, 640)[:, anchors] = True
    else:
        mask[anchors] = True
    repairs = int((mask & actual).sum())
    categories[name] = {'text_repair': repairs, 'circuit_and_native_accommodation': int(actual.sum()) - repairs}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', readerpath, querypath] +
            [ROOT / f'artifacts/{n}.pt' for n in names + [word_name]],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
print(json.dumps({'baseline_cost': old_cost, 'chosen': trials[best], 'actual_margins': actual_margins}, indent=2), flush=True)
