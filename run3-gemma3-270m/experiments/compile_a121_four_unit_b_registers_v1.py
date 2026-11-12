"""Fold duplicate B-reference GLUs, reducing the amplifier from five to four.

The native-value reader has identical constant and plus-reference Up rows.
Their common gate permits exact Down-column addition. Four newly placed units
then receive the unchanged four functions. Each native repair starts from
original HOST targets, with all post gains fixed. No answers or replay occur.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from native_fixed_gain_reconstruction_v2 import NativeFixedGainReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a118v1_smaller_grouped_pool/delta.pt'
np = ROOT / 'artifacts/a88v1_native_mlp2.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
n = torch.load(np, weights_only=True, mmap=True)
stage = c['native_b_stage']
units0 = stage['units'].long()
donors = stage['donors'].long()
assert len(units0) == 5 and len(donors) == 512
prefix = 'model.layers.2.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
G0, U0, W0 = ch[gn][units0], ch[un][units0], ch[dn][:, units0]
assert torch.equal(G0, G0[:1].expand_as(G0))
assert torch.equal(U0[0], U0[4]), 'B constant/reference functions must be identical'
Gprog, Uprog = G0[:4].clone(), U0[:4].clone()
Wprog = W0[:, :4].double().clone(); Wprog[:, 0] += W0[:, 4].double()
assert Wprog.abs().max() <= 1e6
# Confirm the diagnostic's input provenance without replaying any layer.
capture_base = ROOT / 'experiments/a88v1_native_mlp2/delta.pt'
assert capture_base.exists()
reference = expand(h, capture_base)
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.')) or (
            name.startswith(prefix) and ('.self_attn.' in name or name.endswith('input_layernorm.weight') or name.endswith('post_attention_layernorm.weight') or name.endswith('pre_feedforward_layernorm.weight'))):
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
for name in (gn, un, dn, pn):
    assert torch.equal(reference.get(name, h[name]), h[name]), name
assert n['checkpoint_sha256'] == sha(capture_base)

gamma = 1 + ch[pn].double()
pool = (gamma.abs() > 1e5).nonzero().flatten()
model = NativeFixedGainReconstruction(n['captures']['2.features'], h[dn], 1 + h[pn].double(), ch[dn], gamma, units0)
baseline = model.quality(ch[dn], gamma, units0)
available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
D = model.FH[:model.nt, donors] / model.hr[:model.nt]
F = model.FH[:model.nt, available] / model.hr[:model.nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
C = Z.T @ Z / len(Z); C.diagonal().add_(.001)
fit = torch.linalg.solve(C, Z.T @ F / len(Z)) / sd[:, None]
conditional = (F - D @ fit).square().mean(0)
post = conditional * (model.WH[:, available] * model.gh[:, None]).square().sum(0)
raw = conditional * model.WH[:, available].square().sum(0)
post /= post.sum(); raw /= raw.sum()
trials = []; best = None; seen = set()
for mass in (0., .01, .1, 1., 10.):
    units = available[(post + mass * raw).argsort()[:4]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'mass': mass, 'duplicate_unit_set': True}); continue
    seen.add(key)
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    assert torch.equal(parameter, ch[pn])
    row = {'mass': mass, 'units': units.tolist(), 'repair': repair}
    trials.append(row); print(json.dumps(row), flush=True)
    q = repair['selected']['validation_native_branch_rms']
    if best is None or q < best[0]: best = q, W, units.clone(), row
report = {'programme_count_before': 5, 'programme_count_after': 4,
          'folded_old_positions': [0, 4], 'identical_G_and_Up_verified': True,
          'raw_importance_mass_trials': [0,.01,.1,1,10], 'native_donor_count': 512,
          'baseline': baseline, 'trials': trials, 'selected': None if best is None else best[-1],
          'all_post_gains_fixed': True, 'no_answers_gradients_or_activation_replay': True}
out = ROOT / 'experiments/a121v1_four_unit_b_registers/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'four_unit_b_registers.json', report)
assert best is not None and best[0] < .98 * baseline['validation_native_branch_rms'], 'Less than two-percent native branch improvement'
_, W, units, _ = best
G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Gprog, Uprog
ch[gn], ch[un], ch[dn] = G, U, W
report['programme_units'] = units.tolist()
c['four_unit_b_registers_a121'] = report
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
for name in (gn, un): categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
changed = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(changed); mask[:, units] = False
repair_count = int((changed & mask).sum())
categories[dn] = {'native_text_repair': repair_count, 'arithmetic_programme': int(changed.sum()) - repair_count}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/native_fixed_gain_reconstruction_v2.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np, capture_base],
     arguments={}, description=__doc__, extra=report, categories=categories)
