"""Compile three common-gated nearest-digit features with a raw norm anchor.

Geometry is the fixed literal A86 construction:128rows, class bias32, anchor
amplitude2. Native feature placement uses five fixed conditional-cost masses;
every native repair starts from HOST and the same unlabeled calibration target.
Programme features are copies or linear combinations of recorded functions.
No new GELU is evaluated, and no sums, gradients or iterative fits are used.
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
base = ROOT / 'experiments/a84v1_hinge_output128/delta.pt'
np = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a84v1_gate_features.pt'
nfp = ROOT / 'artifacts/a85v1_native_format_units.pt'
gp = ROOT / 'artifacts/a86_anchored_linear_emitter_v1.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
n = torch.load(np, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
native_fmt = torch.load(nfp, weights_only=True, mmap=True)
geometry = torch.load(gp, weights_only=True)['candidates']['128_2']
assert geometry['report']['wrong_current_register_outputs'] == 0
pool = geometry['coordinates'].long()
source_units = c['units'].long()
available = c['previous_threshold_units'].long()
assert len(source_units) == 22 and len(available) == 31
assert torch.equal(fmt['inputs'], native_fmt['inputs'])
assert native_fmt['request']['capture']['17.native_units']['channels'] == available.tolist()
donors = c['native_donor_repairs']['17']['donors'].long()
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
Gprogramme = torch.cat([ch[gn][source_units[:3]], ch[gn][source_units[11:]]]).clone()
Uprogramme = torch.cat([ch[un][source_units[:3]], ch[un][source_units[11:]]]).clone()
assert torch.equal(Gprogramme[0], Gprogramme[1]) and torch.equal(Gprogramme[0], Gprogramme[2])
Uprogramme[1] = (ch[un][source_units[1]].double() - 4.5 * ch[un][source_units[0]].double()).float()
og = geometry['report']['raw_output_gain']
amplitude = geometry['report']['anchor_amplitude']
anchor = geometry['report']['anchor_coordinate']
cancel_scale = geometry['report']['cancellation_scale'] / max(1., amplitude)
Wprogramme = torch.zeros(640, 14, dtype=torch.float64)
Wprogramme[pool, 0] = og * geometry['constant'].double()
Wprogramme[pool, 1] = og * geometry['linear'].double()
Wprogramme[pool, 2] = og * geometry['newline'].double()
Wprogramme[anchor, 0] = og * amplitude
Wprogramme[anchor, 2] = og * amplitude
Wprogramme[:, 3:] = ch[dn][:, source_units[11:]].double() * cancel_scale
model = NativeFinalReconstruction(n['captures']['17.features'], h[dn], 1 + h[pn].double())
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
print(json.dumps({'baseline': baseline}), flush=True)
for mass in (0., .01, .1, 1., 10.):
    units = available[(post_cost + mass * raw_cost).argsort()[:14]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'raw_importance_mass': mass, 'duplicate_unit_set': True, 'units': units.tolist()})
        continue
    seen.add(key)
    W, parameter, repair_report = model.reconstruct(units, Wprogramme, pool, donors)
    row = {'raw_importance_mass': mass, 'units': units.tolist(), 'repair': repair_report}
    trials.append(row)
    print(json.dumps(row), flush=True)
    q = repair_report['selected']['validation_native_branch_rms']
    if best is None or q < best[0]:
        best = q, W, parameter, units.clone(), row
assert best is not None
q, W, parameter, units, selected = best
G, U = h[gn].clone(), h[un].clone()
G[units], U[units] = Gprogramme, Uprogramme
PF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048).clone()
old_fields = PF[:, source_units].clone()
PF[:, available] = native_fmt['captures']['17.native_units'][:, 8:13].double().reshape(-1, 31)
new_fields = torch.cat([old_fields[:, :3], old_fields[:, 11:]], dim=1)
# A common recorded gate multiplies each linear Up row. This identity changes
# only the Up projection; the conservative float-rounding caveat is metered.
new_fields[:, 1] = old_fields[:, 1] - 4.5 * old_fields[:, 0]
PF[:, units] = new_fields
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
gain = 1 + parameter.double()
R = PF @ W.double().T
output = before + R / (R.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
output /= (output.square().mean(1, keepdim=True) + 1e-6).sqrt()
factor = c['piece']['source_flag_scale'] / before[:, c['piece']['source_flag_coordinate']]
digit = before[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = before[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
labels = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
ids = torch.tensor(c['piece']['token_ids'])
expected = ids[labels]
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
desired = (output * fg * E[expected].double()).sum(1)
other = torch.full_like(desired, -torch.inf)
for start in range(0, len(E), 4096):
    logits = output @ (E[start:start + 4096].double() * fg).T
    local = expected - start
    rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
    logits[rows, local[rows]] = -torch.inf
    other = torch.maximum(other, logits.amax(1))
margin = desired - other
word_gate_max = float((n['captures']['17.pre'].double().reshape(-1, 640) @ G[units].double().T).max())
assert word_gate_max < -31.9
report = {'literal_geometry': geometry['report'], 'programme_units': 14,
          'emitter_units': 3, 'cancellation_units': 11,
          'raw_importance_mass_trials': [0, .01, .1, 1, 10],
          'conditional_cost_donor_count': 512, 'conditional_cost_ridge': .001,
          'baseline': baseline, 'trials': trials, 'selected': selected,
          'minimum_static_current_register_margin': float(margin.min()),
          'wrong_static_current_register_outputs': int((margin <= 0).sum()),
          'maximum_word_programme_gate': word_gate_max,
          'anchor_native_post_gain': float(gain[anchor]),
          'format_feature_caveat': 'The one new Up row is a linear combination of recorded common-gated features; float32 projection/product rounding is checked by the official forward.',
          'no_activation_replay_or_answer_targets': True}
print(json.dumps({'report': report}), flush=True)
assert (margin > 0).all(), 'Anchored linear emitter fails full component readout'
ch[gn], ch[un], ch[dn], ch[pn] = G, U, W, parameter
c['units'] = units
c['anchored_linear_output'] = report
c['previous_hinge_emitter'] = c.pop('hinge_emitter')
c['output_pool'] = pool
c['output_directions'] = torch.stack([geometry[k] for k in ('constant', 'linear', 'newline')], dim=1)
c['piece'].update(selected_units=units.tolist(), emitter_kind='three-feature anchored nearest-digit',
                  output_gain=og, raw_anchor_coordinate=anchor, raw_anchor_amplitude=amplitude)
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'anchored_linear_output.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(actual)
mask[:, units] = False
nrepair = int((actual & mask).sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output_and_cancellation': int(actual.sum()) - nrepair}
ordinary = torch.ones(640, dtype=torch.bool)
ordinary[pool] = False
actual = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'native_text_repair': int(actual[ordinary].sum()), 'arithmetic_writer_gains': int(actual[pool].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py',
                         ROOT / 'experiments/analyze_a86_anchored_linear_emitter_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', np, fp, nfp, gp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
