"""Restore native final-MLP normalization after compact-emitter rebuilding.

Original HOST branch and raw energy are the only word targets. One fixed
HOST-normalizer strong-row solve and independent squared-row energy solves
reset ordinary gains to native values. Energy scaling changes native Down
columns only; literal programme Down columns and strong gains are fixed.
No answer fitting, gradients, activation replay or iterative fitting occurs.
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
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, ELIGIBLE

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a98v1_pooled_gate_bounds/delta.pt'
np = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a98v1_syntax_stress.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
n, fmt = [torch.load(path, weights_only=True, mmap=True) for path in (np, fp)]
dn, pn = 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight'
Wbase, gb = ch[dn].double(), 1 + ch[pn].double()
WH, gh = h[dn].double(), 1 + h[pn].double()
units = c['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
pool = (gb.abs() > 1e5).nonzero().flatten()
ordinary = torch.ones(640, dtype=torch.bool)
ordinary[pool] = False
assert len(pool) == 128 and torch.equal(gb[pool], torch.full_like(gb[pool], 1e6))
FH = n['captures']['17.features'].double().reshape(-1, 2048)
RH = FH @ WH.T
HE = RH.square().mean(-1) + 1e-6
hr = HE.sqrt()[:, None]
target = RH / hr * gh
F = FH.clone()
F[:, units] = 0
nt = 384 * 64
def quality(W, gain):
    R = F[nt:] @ W.double().T
    energy = R.square().mean(-1) + 1e-6
    error = R / energy.sqrt()[:, None] * gain.double() - target[nt:]
    return {'validation_native_branch_rms': float(error.square().mean().sqrt()),
            'validation_relative_raw_energy_rms': float((energy / HE[nt:] - 1).square().mean().sqrt())}
baseline = quality(ch[dn], gb)
# One original HOST-reference solve, independent of all energy-row trials.
X = FH[:nt, native] / hr[:nt]
std = X.square().mean(0).sqrt().clamp_min(1e-12)
Z = X / std
C = Z.T @ Z / nt
ev, evec = torch.linalg.eigh((C + C.T) / 2)
ev = ev.clamp_min(0)
rhs = evec.T @ (Z.T @ (target[:nt, pool] / gb[pool]) / nt)
strong_ridge = .0001
fit = (evec @ (rhs / (ev[:, None] + strong_ridge))) / std[:, None]
W0 = Wbase.clone()
W0[pool[:, None], native[None]] = fit.T
W0 = W0.float().double()
raw = F @ W0.T
energy = raw.square().mean(-1) + 1e-6
design = raw.square() / (640 * HE[:, None])
energy_target = 1 - energy / HE
diag = design[:nt].square().mean(0)
cross = design[:nt].T @ energy_target[:nt] / nt
priority = cross.square() / diag.clamp_min(1e-30)
priority[~ordinary] = -torch.inf
order = priority.argsort(descending=True)
PF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
base_format_raw = PF @ Wbase.T
base_format_post = base_format_raw / (base_format_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gb
trials, best = [], None
print(json.dumps({'baseline': baseline, 'energy_row_order': order[:16].tolist()}), flush=True)
for count in (1, 2, 4, 8, 16):
    rows = order[:count]
    assert ordinary[rows].all()
    sd = diag[rows].sqrt().clamp_min(1e-30)
    D = design[:nt, rows] / sd
    cov, rhs = D.T @ D / nt, D.T @ energy_target[:nt] / nt
    for ridge in (1e-6, 1e-4, .01, 1., 100.):
        R = cov.clone()
        R.diagonal().add_(ridge)
        beta = torch.linalg.solve(R, rhs) / sd
        row = {'row_count': count, 'rows': rows.tolist(), 'ridge': ridge, 'eligible': False}
        if (1 + beta).min() <= 0:
            row['reason'] = 'Nonpositive squared energy-row scale'
            trials.append(row)
            print(json.dumps(row), flush=True)
            continue
        scales = (1 + beta).sqrt()
        W = W0.clone()
        W[rows[:, None], native[None]] *= scales[:, None]
        W = W.float()
        gain = gh.clone()
        gain[pool] = gb[pool]
        gain[rows] /= scales
        parameter = (gain - 1).float()
        gain = 1 + parameter.double()
        assert torch.equal(W[:, units], ch[dn][:, units])
        assert torch.equal(parameter[pool], ch[pn][pool])
        metrics = quality(W, gain)
        FR = PF @ W.double().T
        post = FR / (FR.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gain
        changed = bm['changed_scalars'] - bm['per_tensor'][dn] - bm['per_tensor'][pn]
        changed += int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
        changed += int((parameter.view(torch.int32) != h[pn].view(torch.int32)).sum())
        row.update(raw_row_scales=scales.tolist(), **metrics,
                   maximum_format_post_change=float((post - base_format_post).abs().max()),
                   changed_scalars=changed, maximum_down_weight=float(W.abs().max()))
        row['eligible'] = bool(W.isfinite().all() and parameter.isfinite().all() and W.abs().max() <= 1e6
                               and parameter.abs().max() <= 1e6 and changed <= int(.05 * ELIGIBLE)
                               and row['maximum_format_post_change'] < 10)
        trials.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and (best is None or metrics['validation_native_branch_rms'] < best[0]):
            best = metrics['validation_native_branch_rms'], W.clone(), parameter.clone(), row, rows.clone()
assert best is not None
q, W, parameter, selected, rows = best
assert q < .99 * baseline['validation_native_branch_rms'], 'Less than one percent local improvement; do not meter'
gain = 1 + parameter.double()
R = PF @ W.double().T
output = before + R / (R.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gain
output /= (output.square().mean(-1, keepdim=True) + 1e-6).sqrt()
factor = c['piece']['source_flag_scale'] / before[:, c['piece']['source_flag_coordinate']]
digit = before[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = before[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
labels = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
expected = torch.tensor(c['piece']['token_ids'])[labels]
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
desired = (output * fg * E[expected].double()).sum(-1)
other = torch.full_like(desired, -torch.inf)
for start in range(0, len(E), 4096):
    logits = output @ (E[start:start + 4096].double() * fg).T
    local = expected - start
    valid = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
    logits[valid, local[valid]] = -torch.inf
    other = torch.maximum(other, logits.amax(-1))
margin = desired - other
report = {'baseline': baseline, 'row_count_trials': [1, 2, 4, 8, 16],
          'energy_ridge_trials': [1e-6, 1e-4, .01, 1, 100], 'strong_original_host_ridge': strong_ridge,
          'strong_reference': 'frozen HOST native raw normalizer at the exact current input',
          'ordinary_gains_reset_to_HOST_with_inverse_native_energy_row_compensation': True,
          'programme_Down_columns_and_strong_gains_bit_identical': True,
          'trials': trials, 'selected': selected, 'minimum_static_current_register_margin': float(margin.min()),
          'wrong_static_current_register_outputs': int((margin <= 0).sum()),
          'no_answers_gradients_activation_replay_or_iterative_fit': True}
print(json.dumps({'selected': selected, 'minimum_margin': float(margin.min())}), flush=True)
assert (margin > 0).all(), 'Native energy repair fails a recorded literal register readout'
ch[dn], ch[pn] = W, parameter
c['native_final_energy_a102'] = report
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'native_energy.json', report)
categories = dict(bm['categories'])
actual = W.view(torch.int32) != h[dn].view(torch.int32)
native_count = int(actual[:, native].sum())
categories[dn] = {'native_text_reconstruction_and_energy': native_count, 'arithmetic_programme': int(actual.sum()) - native_count}
actual_gain = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'arithmetic_strong_gains': int(actual_gain[pool].sum()), 'native_energy_accommodation': int(actual_gain[ordinary].sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
