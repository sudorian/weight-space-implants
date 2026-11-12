"""Relocate layer6 programme units by conditional native-feature cost.

All 18 literal programme functions and their Down columns are permuted
exactly. Candidate units come from the native non-donor pool, using actual
current-input HOST features. Every distinct placement gets one independent
original-HOST native reconstruction. Five declared raw-importance masses
select by held-back native branch error. No arithmetic targets or activation
replay enter the construction; arithmetic must be checked by the meter.
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
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a90v1_grouped_affine/delta.pt'
np = ROOT / 'artifacts/a92v1_native_mlp6.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
n = torch.load(np, weights_only=True, mmap=True)
prefix = 'model.layers.6.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
source_units = ((ch[gn] != h[gn]).any(1) | (ch[un] != h[un]).any(1)).nonzero().flatten()
assert len(source_units) == 18
Gprogramme, Uprogramme = ch[gn][source_units].clone(), ch[un][source_units].clone()
Wprogramme = ch[dn][:, source_units].clone()
writer = Wprogramme.ne(0).any(1)
pool = writer.nonzero().flatten()
assert torch.equal(1 + ch[pn][pool], torch.full_like(ch[pn][pool], 1e6))
donors = c['native_donor_repairs']['6']['donors'].long()
assert len(donors) == 512 and not torch.isin(source_units, donors).any()
available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
model = NativeFinalReconstruction(n['captures']['6.features'], h[dn], 1 + h[pn].double())
baseline = model.quality(ch[dn], 1 + ch[pn].double(), source_units)
nt = model.nt
D = model.FH[:nt, donors] / model.hr[:nt]
P = model.FH[:nt, available] / model.hr[:nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12)
Z = D / sd
C = Z.T @ Z / len(Z)
C.diagonal().add_(.001)
fit = torch.linalg.solve(C, Z.T @ P / len(Z)) / sd[:, None]
conditional = (P - D @ fit).square().mean(0)
post_cost = conditional * (model.WH[:, available] * model.gh[:, None]).square().sum(0)
raw_cost = conditional * model.WH[:, available].square().sum(0)
post_cost /= post_cost.sum()
raw_cost /= raw_cost.sum()
trials, seen, best = [], set(), None
print(json.dumps({'baseline': baseline, 'writer_coordinates': pool.tolist()}), flush=True)
for mass in (0., .01, .1, 1., 10.):
    units = available[(post_cost + mass * raw_cost).argsort()[:len(source_units)]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'raw_importance_mass': mass, 'duplicate_unit_set': True, 'units': units.tolist()})
        continue
    seen.add(key)
    W, parameter, repair_report = model.reconstruct(units, Wprogramme, pool, donors)
    assert torch.equal(W[:, units], Wprogramme)
    assert torch.equal(parameter[pool], ch[pn][pool])
    row = {'raw_importance_mass': mass, 'units': units.tolist(), 'repair': repair_report}
    trials.append(row)
    print(json.dumps(row), flush=True)
    q = repair_report['selected']['validation_native_branch_rms']
    if best is None or q < best[0]:
        best = q, W, parameter, units.clone(), row
assert best is not None
q, W, parameter, units, selected = best
assert q < .99 * baseline['validation_native_branch_rms'], 'Less than one percent local improvement; do not meter'
G, U = h[gn].clone(), h[un].clone()
G[units], U[units] = Gprogramme, Uprogramme
word_gate_max = float((n['captures']['6.pre'].double().reshape(-1, 640) @ G[units].double().T).max())
assert word_gate_max < -8
report = {'layer': 6, 'previous_programme_units': source_units.tolist(), 'programme_units': units.tolist(),
          'native_candidate_pool': 'all 1536 native units outside the fixed 512 donor pool',
          'writer_coordinates': pool.tolist(), 'raw_importance_mass_trials': [0, .01, .1, 1, 10],
          'conditional_cost_donor_count': 512, 'conditional_cost_ridge': .001,
          'baseline': baseline, 'trials': trials, 'selected': selected,
          'maximum_word_programme_gate': word_gate_max,
          'all_programme_functions_Down_columns_and_writer_gains_bit_identical_after_permutation': True,
          'no_format_readout_claim_before_official_meter': True,
          'old_layer6_unit_indices_in_inherited_circuit_metadata_superseded_by_this_mapping': True,
          'no_activation_replay_or_arithmetic_answer_targets': True}
ch[gn], ch[un], ch[dn], ch[pn] = G, U, W, parameter
c['layer6_native_unit_reassignment'] = report
c['previous_native_donor_repair_layer6'] = c['native_donor_repairs']['6']
c['native_donor_repairs']['6'] = {'donors': donors, 'programme_units': units, 'report': selected['repair']}
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'native_unit_reassignment.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
native = torch.ones(2048, dtype=torch.bool)
native[units] = False
repair_count = int(actual[:, native].sum())
categories[dn] = {'native_text_repair': repair_count, 'arithmetic_programme': int(actual.sum()) - repair_count}
actual_gamma = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'arithmetic_writer_gains': int(actual_gamma[writer].sum()), 'native_gain_accommodation': int(actual_gamma[~writer].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
