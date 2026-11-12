"""Suppress small non-newline register noise in the three-feature emitter.

An exact affine change of basis uses constant, digit and newline functions.
The newline function is then replaced by a literal saturated half-threshold:
2*max(newline/flag-.5,0). Its gate is built only from calibration format/word
states. Native Down coefficients and gains are unchanged. No answer fitting,
new activation evaluation, gradients or iterative fitting occurs.
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
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a86v1_anchored_linear/delta.pt'
np = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a84v1_gate_features.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'rectified_newline' not in c
n = torch.load(np, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
units = c['units'].long()
assert len(units) == 14
prefix = 'model.layers.17.'
gn, un, dn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')]
G0 = ch[gn][units[0]].double()
assert torch.equal(ch[gn][units[0]], ch[gn][units[1]]) and torch.equal(ch[gn][units[0]], ch[gn][units[2]])
oldU = ch[un][units[:3]].double()
flag = oldU[0] + oldU[2]
digit = oldU[1] + 4.5 * oldU[0]
newline = oldU[2]
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
negative = torch.cat([n['captures']['17.pre'].double().reshape(-1, 640),
                      fmt['captures']['17.pre'][:, :8].double().reshape(-1, 640)])
pf = P @ flag
assert pf.min() > 0 and (P @ G0).min() > 16
fraction = (P @ newline) / pf
is_newline = fraction > .5
common_scale = float(((P @ G0) / pf).median())
off = common_scale * flag - G0
no, po = negative @ off, P @ off
assert no.min() > 0
literal = newline - .5 * flag
trials, best = [], None
for sharp in (16., 64., 256., 1024., 4096.):
    M = float(((sharp * (negative @ literal) + 512.) / no).max().clamp_min(0))
    gate = (sharp * literal - M * off).float()
    up = (2 / sharp * G0).float()
    neg = negative @ gate.double()
    positive = P @ gate.double()
    signed = positive * torch.where(is_newline, 1., -1.)
    shift = M * po / (sharp * pf)
    row = {'sharpness': sharp, 'off_multiplier': M,
           'maximum_negative_gate': float(neg.max()),
           'minimum_signed_format_gate': float(signed.min()),
           'maximum_normalized_newline_gate_distortion': float((2 * shift).abs().max()),
           'maximum_gate_weight': float(gate.abs().max()), 'maximum_Up_weight': float(up.abs().max())}
    row['eligible'] = bool(gate.isfinite().all() and up.isfinite().all()
                           and gate.abs().max() <= 1e6 and up.abs().max() <= 1e6
                           and neg.max() < -511.9 and signed.min() > 16)
    trials.append(row)
    print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or row['maximum_normalized_newline_gate_distortion'] < best[0]):
        best = row['maximum_normalized_newline_gate_distortion'], gate, up, row
assert best is not None
_, gate, up, selected = best
G, U, W = ch[gn].clone(), ch[un].clone(), ch[dn].clone()
G[units[2]] = gate
U[units[0]] = flag.float()
U[units[1]] = digit.float()
U[units[2]] = up
oldD = ch[dn][:, units[:3]].double()
W[:, units[0]] = (oldD[:, 0] - 4.5 * oldD[:, 1]).float()
W[:, units[1]] = oldD[:, 1].float()
W[:, units[2]] = (oldD[:, 2] - oldD[:, 0] + 4.5 * oldD[:, 1]).float()
native = torch.ones(2048, dtype=torch.bool)
native[units] = False
for new, old in ((G, ch[gn]), (U, ch[un])):
    assert torch.equal(new[native], old[native])
assert torch.equal(W[:, native], ch[dn][:, native])
assert torch.equal(W[:, units[3:]], ch[dn][:, units[3:]])
assert G.abs().max() <= 1e6 and U.abs().max() <= 1e6 and W.abs().max() <= 1e6
word_maximum = float((n['captures']['17.pre'].double().reshape(-1, 640) @ G[units].double().T).max())
assert word_maximum < -511.9
report = {'sharpness_trials': [16, 64, 256, 1024, 4096], 'negative_gate_margin': 512.,
          'literal_newline_threshold': .5, 'positive_slope': 2.,
          'common_scale': common_scale, 'trials': trials, 'selected': selected,
          'maximum_word_programme_gate': word_maximum,
          'minimum_digit_phase_newline_fraction': float(fraction[~is_newline].min()),
          'maximum_digit_phase_newline_fraction': float(fraction[~is_newline].max()),
          'newline_phase_fraction_min': float(fraction[is_newline].min()),
          'newline_phase_fraction_max': float(fraction[is_newline].max()),
          'all_native_weights_gains_and_cancellation_functions_bit_identical': True,
          'format_readout_requires_official_forward': True,
          'no_failure_capture_answer_targets_or_activation_replay': True}
ch[gn], ch[un], ch[dn] = G, U, W
c['rectified_newline'] = report
c['piece']['emitter_kind'] = 'anchored nearest-digit with rectified newline'
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'rectified_newline.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
nrepair = int(actual[:, native].sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output_and_cancellation': int(actual.sum()) - nrepair}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
