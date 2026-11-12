"""Choose twenty-two cheaper native units for the unchanged hinge functions.

The pool is the original thirty-one reused units. Conditional feature costs
come from one HOST-target donor solve on unlabeled current calibration data.
Programme G/Up/Down functions are permuted exactly. Native Down/gains are
independent original-HOST reconstructions for five fixed importance masses.
Recorded HOST features plus permuted actual programme features give exact
static word/format components, without activation replay.
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
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a84v1_hinge_output128/delta.pt'
np = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a84v1_gate_features.pt'
nfp = ROOT / 'artifacts/a85v1_native_format_units.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'compact_unit_selection' not in c
n = torch.load(np, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
native_fmt = torch.load(nfp, weights_only=True, mmap=True)
assert torch.equal(fmt['inputs'], native_fmt['inputs'])
units = c['units'].long()
available = c['previous_threshold_units'].long()
assert len(units) == 22 and len(available) == 31
assert native_fmt['request']['capture']['17.native_units']['channels'] == available.tolist()
pool = c['output_pool'].long()
donors = c['native_donor_repairs']['17']['donors'].long()
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
Gprogram = ch[gn][units].clone()
Uprogram = ch[un][units].clone()
Wprogram = ch[dn][:, units].clone()
model = NativeFinalReconstruction(n['captures']['17.features'], h[dn], 1 + h[pn].double())
baseline = model.quality(ch[dn], 1 + ch[pn].double(), units)
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
Pactual = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
native_fields = native_fmt['captures']['17.native_units'][:, 8:13].double().reshape(-1, 31)
programme_fields = Pactual[:, units].clone()
Pnative = Pactual.clone()
Pnative[:, available] = native_fields
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
trials, seen, best = [], set(), None
print(json.dumps({'baseline': baseline}), flush=True)
for mass in (0., .01, .1, 1., 10.):
    chosen_units = available[(post_cost + mass * raw_cost).argsort()[:len(units)]]
    key = tuple(sorted(chosen_units.tolist()))
    if key in seen:
        trials.append({'raw_importance_mass': mass, 'duplicate_unit_set': True, 'units': chosen_units.tolist()})
        continue
    seen.add(key)
    W, parameter, repair_report = model.reconstruct(chosen_units, Wprogram, pool, donors)
    row = {'raw_importance_mass': mass, 'units': chosen_units.tolist(), 'repair': repair_report}
    trials.append(row)
    print(json.dumps(row), flush=True)
    q = repair_report['selected']['validation_native_branch_rms']
    if best is None or q < best[0]:
        best = q, W, parameter, chosen_units.clone(), row
assert best is not None
q, W, parameter, new_units, selected = best
assert q < baseline['validation_native_branch_rms'] - 1e-6, 'Unit relocation does not improve the original native target'
G, U = h[gn].clone(), h[un].clone()
G[new_units], U[new_units] = Gprogram, Uprogram
PF = Pnative.clone()
PF[:, new_units] = programme_fields
gain = 1 + parameter.double()
raw = PF @ W.double().T
out_state = before + raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
out_state /= (out_state.square().mean(1, keepdim=True) + 1e-6).sqrt()
factor = c['piece']['source_flag_scale'] / before[:, c['piece']['source_flag_coordinate']]
digit = before[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = before[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
labels = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
ids = torch.tensor(c['piece']['token_ids'])
expected = ids[labels]
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
desired = (out_state * fg * E[expected].double()).sum(1)
other = torch.full_like(desired, -torch.inf)
for start in range(0, len(E), 4096):
    logits = out_state @ (E[start:start + 4096].double() * fg).T
    local = expected - start
    rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
    logits[rows, local[rows]] = -torch.inf
    other = torch.maximum(other, logits.amax(1))
margin = desired - other
assert (margin > 0).all()
word_gate_max = float((n['captures']['17.pre'].double().reshape(-1, 640) @ G[new_units].double().T).max())
assert word_gate_max < -31.9
report = {'raw_importance_mass_trials': [0, .01, .1, 1, 10],
          'conditional_cost_donor_count': 512, 'conditional_cost_ridge': .001,
          'candidate_native_units': available.tolist(), 'old_programme_units': units.tolist(),
          'baseline': baseline, 'trials': trials, 'selected': selected,
          'minimum_static_current_register_margin': float(margin.min()),
          'maximum_word_programme_gate': word_gate_max,
          'programme_functions_exactly_permuted': True,
          'native_and_programme_features_are_recorded_at_the_same_input': True}
ch[gn], ch[un], ch[dn], ch[pn] = G, U, W, parameter
c['units'] = new_units
c['compact_unit_selection'] = report
c['piece']['selected_units'] = new_units.tolist()
c['hinge_emitter']['new_programme_units'] = new_units.tolist()
c['hinge_emitter']['emitter_units'] = new_units[:11].tolist()
c['hinge_emitter']['cancellation_units'] = new_units[11:].tolist()
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'compact_unit_selection.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(actual)
mask[:, new_units] = False
nrepair = int((actual & mask).sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output_and_cancellation': int(actual.sum()) - nrepair}
ordinary = torch.ones(640, dtype=torch.bool)
ordinary[pool] = False
actual = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'native_text_repair': int(actual[ordinary].sum()), 'arithmetic_writer_gains': int(actual[pool].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np, fp, nfp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
