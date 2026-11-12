"""Build eleven final emitter features with analytically bounded hinge gates.

This is a diagnostic feature checkpoint: its old Down map is intentionally
unchanged. The official meter records the new GELU features before a separate
native-output compiler writes the final Down map. No activation is replayed.
Each hinge uses the existing common gate as its Up factor, making its output
share the exact common normalization factor of the linear emitter features.
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
from tools.common import atomic_torch, atomic_json

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a76v1_pool96/delta.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
wp = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a76v1_syntax_stress.pt'
w = torch.load(wp, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
prefix = 'model.layers.17.'
gn, un = prefix + 'mlp.gate_proj.weight', prefix + 'mlp.up_proj.weight'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
old_units = c['units'].long()
assert len(old_units) == 31
G0 = ch[gn][old_units[0]].double()
assert torch.equal(ch[gn][old_units[0]], ch[gn][old_units[1]])
def coordinate(i):
    row = torch.zeros(640, dtype=torch.float64)
    row[i] = 1 / gp[i]
    return row
flag = coordinate(c['piece']['source_flag_coordinate'])
digit = coordinate(c['piece']['source_digit_coordinate']) * c['piece']['source_flag_scale'] / c['piece']['source_digit_scale']
newline = coordinate(c['piece']['source_newline_coordinate']) * c['piece']['source_flag_scale'] / c['piece']['source_newline_scale']
valid = flag - newline
alpha = float((valid * ch[un][old_units[0]].double()).sum() / valid.square().sum())
assert (alpha * valid - ch[un][old_units[0]].double()).abs().max() < 1e-7
assert (alpha * newline - ch[un][old_units[1]].double()).abs().max() < 1e-7
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
negative = torch.cat([w['captures']['17.pre'].double().reshape(-1, 640),
                      fmt['captures']['17.pre'][:, :8].double().reshape(-1, 640)])
pf = P @ flag
assert pf.min() > 0 and (P @ G0).min() > 16
common_scale = float(((P @ G0) / pf).median())
off = common_scale * flag - G0
no, po = negative @ off, P @ off
assert no.min() > 0
literal = digit[:, None] - flag[:, None] * torch.arange(1, 9, dtype=torch.float64)[None]
negative_literal = negative @ literal
new_units = torch.cat([old_units[:11], old_units[20:]])
trials, best = [], None
for sharp in (16., 64., 256., 1024., 4096.):
    M = ((sharp * negative_literal + 32.) / no[:, None]).amax(0).clamp_min(0)
    gates = sharp * literal - off[:, None] * M[None]
    G = h[gn].clone()
    U = h[un].clone()
    G[new_units[:3]] = G0.float()
    U[new_units[0]] = (alpha * valid).float()
    U[new_units[1]] = (alpha * digit).float()
    U[new_units[2]] = (alpha * newline).float()
    for j in range(8):
        G[new_units[3 + j]] = gates[:, j].float()
        U[new_units[3 + j]] = (alpha / sharp * G0).float()
    G[old_units[20:]] = ch[gn][old_units[20:]]
    U[old_units[20:]] = ch[un][old_units[20:]]
    max_gate = float((negative @ G[new_units].double().T).max())
    # ReLU is 1-Lipschitz and |GELU(x)-ReLU(x)| < .25 for either installed
    # tanh GELU or exact GELU. This is a uniform bound, not an activation replay.
    shift = po[:, None] * M[None] / (sharp * pf[:, None])
    error_bound = shift.abs() + .25 / (sharp * pf[:, None])
    row = {'sharpness': sharp, 'maximum_negative_gate': max_gate,
           'maximum_hinge_scalar_error_bound': float(error_bound.max()),
           'maximum_gate_weight': float(G[new_units].abs().max()),
           'maximum_Up_weight': float(U[new_units].abs().max()),
           'off_multipliers': M.tolist()}
    row['eligible'] = bool(G.isfinite().all() and U.isfinite().all()
                           and G.abs().max() <= 1e6 and U.abs().max() <= 1e6 and max_gate < -31.9)
    trials.append(row)
    print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or row['maximum_hinge_scalar_error_bound'] < best[0]):
        best = row['maximum_hinge_scalar_error_bound'], G, U, row
assert best is not None
_, G, U, chosen = best
report = {'diagnostic_feature_checkpoint': True, 'sharpness_trials': [16, 64, 256, 1024, 4096],
          'negative_margin': 32., 'common_scale': common_scale, 'common_Up_scale': alpha,
          'minimum_negative_off': float(no.min()), 'trials': trials, 'selected': chosen,
          'old_programme_units': old_units.tolist(), 'new_programme_units': new_units.tolist(),
          'emitter_units': new_units[:11].tolist(), 'cancellation_units': old_units[20:].tolist(),
          'hinge_knots': list(range(1, 9)), 'GELU_ReLU_uniform_bound': .25,
          'no_GELU_attention_model_replay_or_answer_targets': True}
ch[gn], ch[un] = G, U
c['hinge_emitter'] = report
c['previous_threshold_units'] = old_units
c['units'] = new_units
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'hinge_gates.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
