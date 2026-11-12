"""Finish the eleven-feature emitter from actual recorded new GELU features.

Native output reconstruction starts from HOST at the exact current input,
with one lost-feature donor solve and independent original-target strong-row
ridge solves. Literal digit directions and eight second differences determine
all arithmetic Down columns. No answers, activation replay, gradients or
iterative residual fitting are used.
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
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--pool-count', type=int, choices=[32, 48, 64, 96, 128], default=128)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a84v1_hinge_features/delta.pt'
original_path = ROOT / 'experiments/a76v1_pool96/delta.pt'
np = ROOT / 'artifacts/a85v1_native_word.pt'
fp = ROOT / 'artifacts/a84v1_gate_features.pt'
wp = ROOT / 'artifacts/a83v1_word.pt'
gp = ROOT / 'artifacts/a63_sparse_output_geometry_v1.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
native = torch.load(np, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
word = torch.load(wp, weights_only=True, mmap=True)
geo = torch.load(gp, weights_only=True)
assert fmt['checkpoint_sha256'] == sha(base)
assert torch.equal(native['inputs'], word['inputs'])
assert torch.equal(native['captures']['17.pre'], word['captures']['17.pre'])
assert torch.equal(native['captures']['17.raw'], word['captures']['17.raw'])
program = c['units'].long()
old_program = c['previous_threshold_units'].long()
assert len(program) == 22 and len(old_program) == 31
geometry = geo['candidates'][f'2_{a.pool_count}']
assert geometry['report']['all_correct']
pool = geometry['coordinates'].long()
directions = geometry['directions'].double()
assert set(c['piece']['cancellation_coordinates']) <= set(pool.tolist())
pre = 'model.layers.17.'
dn, pn = pre + 'mlp.down_proj.weight', pre + 'post_feedforward_layernorm.weight'
WH = h[dn].double()
gh = 1 + h[pn].double()
Wbase = ch[dn].double()
gb = 1 + ch[pn].double()
FH = native['captures']['17.features'].double().reshape(-1, 2048)
F = FH.clone()
F[:, program] = 0
oldF = FH.clone()
oldF[:, old_program] = 0
RH = FH @ WH.T
hr = (RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
target = RH / hr * gh
current_raw_check = oldF @ Wbase.T
captured_current_raw = word['captures']['17.down'].double().reshape(-1, 640)
native_input_reconstruction_error = float((current_raw_check - captured_current_raw).square().mean().sqrt() / captured_current_raw.square().mean().sqrt())
assert native_input_reconstruction_error < 1e-5
word_pre = native['captures']['17.pre'].double().reshape(-1, 640)
maximum_program_gate = float((word_pre @ ch[pre + 'mlp.gate_proj.weight'][program].double().T).max())
assert maximum_program_gate < -31.9
nt = 384 * 64
donors = c['native_donor_repairs']['17']['donors'].long()
assert len(donors) == 512 and not set(donors.tolist()) & set(program.tolist())
D = FH[:nt, donors] / hr[:nt]
std = D.square().mean(0).sqrt().clamp_min(1e-12)
Z = D / std
C = Z.T @ Z / len(Z)
C.diagonal().add_(.001)
repair = torch.linalg.solve(C, Z.T @ (FH[:nt, program] / hr[:nt]) / len(Z)) / std[:, None]
WN = WH.clone()
WN[:, donors] += WH[:, program] @ repair.T
WN[:, program] = 0
W0 = WN.clone()
W0[pool] *= (gh[pool] / 1e6)[:, None]
gain = gh.clone()
gain[pool] = 1e6
raw = F @ W0.T
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
normalized = raw / rms
ordinary = torch.ones(640, dtype=torch.bool)
ordinary[pool] = False
fitted_gain = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
gain[ordinary] = fitted_gain[ordinary]
gain_parameter = (gain - 1).float()
gain = 1 + gain_parameter.double()
programme_down = torch.zeros(640, len(program), dtype=torch.float64)
og = c['piece']['output_gain']
programme_down[pool, 0] = og * directions[:, 0]
programme_down[pool, 1] = og * (directions[:, 1] - directions[:, 0])
programme_down[pool, 2] = og * directions[:, 10]
second = directions[:, 2:10] - 2 * directions[:, 1:9] + directions[:, :8]
programme_down[pool[:, None], torch.arange(3, 11)[None]] = og * second
programme_down[:, 11:] = Wbase[:, old_program[20:]]
remaining = torch.tensor([i for i in range(2048) if i not in set(program.tolist())])
X = F[:, remaining] / rms
sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-12)
Z = X[:nt] / sd
C = Z.T @ Z / len(Z)
ev, evec = torch.linalg.eigh((C + C.T) / 2)
ev = ev.clamp_min(0)
rhs = evec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / len(Z))

def quality(W, gamma, features):
    r = features[nt:] @ W.T
    post = r / (r.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
    error = post - target[nt:]
    return {'validation_native_branch_rms': float(error.square().mean().sqrt()),
            'validation_native_branch_relative_rms': float(error.square().mean().sqrt() / target[nt:].square().mean().sqrt())}

baseline = quality(Wbase, gb, oldF)
trials, best = [], None
for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
    W = W0.clone()
    fit = (evec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
    W[pool[:, None], remaining[None]] = fit.T
    W[:, program] = programme_down
    W = W.float()
    row = {'ridge': ridge, **quality(W.double(), gain, F), 'maximum_down_weight': float(W.abs().max())}
    row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6)
    trials.append(row)
    print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or row['validation_native_branch_rms'] < best[0]):
        best = row['validation_native_branch_rms'], W.clone(), row
assert best is not None
_, W, selected = best
PF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
R = PF @ W.double().T
output = before + R / (R.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
output /= (output.square().mean(1, keepdim=True) + 1e-6).sqrt()
factor = c['piece']['source_flag_scale'] / before[:, c['piece']['source_flag_coordinate']]
digit = before[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = before[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
labels = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
ids = torch.tensor(geo['token_ids'])
expected = ids[labels]
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
desired = (output * fg * E[expected].double()).sum(1)
other = torch.full_like(desired, -torch.inf)
for start in range(0, len(E), 4096):
    scores = output @ (E[start:start + 4096].double() * fg).T
    local = expected - start
    rows = ((local >= 0) & (local < scores.shape[1])).nonzero().flatten()
    scores[rows, local[rows]] = -torch.inf
    other = torch.maximum(other, scores.amax(1))
margin = desired - other
common = PF[:, program[0]] + PF[:, program[2]]
actual_hinges = PF[:, program[3:11]] / common[:, None]
ideal_hinges = (digit[:, None] - torch.arange(1, 9, dtype=torch.float64)[None]).clamp_min(0)
report = {'pool_count': a.pool_count, 'pool_count_trials': [32, 48, 64, 96, 128],
          'programme_units': 22, 'emitter_units': 11, 'cancellation_units': 11,
          'native_donor_count': 512, 'native_donor_ridge': .001,
          'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'baseline': baseline, 'trials': trials, 'selected': selected,
          'minimum_static_full_vocabulary_margin': float(margin.min()),
          'wrong_static_current_register_outputs': int((margin <= 0).sum()),
          'maximum_actual_hinge_scalar_error': float((actual_hinges - ideal_hinges).abs().max()),
          'maximum_actual_word_programme_gate': maximum_program_gate,
          'native_input_reconstruction_relative_rms': native_input_reconstruction_error,
          'native_features_and_new_format_GELU_features_both_exact_meter_captures': True}
print(json.dumps({'report': report}), flush=True)
assert (margin > 0).all(), 'Actual hinge emitter fails current-register component readout'
ch[dn] = W
ch[pn] = gain_parameter
c['hinge_output_reconstruction'] = report
c['output_pool'] = pool
c['output_directions'] = directions
c['piece'].update(output_pool=pool.tolist(), output_pool_size=len(pool),
                  output_geometry=geometry['report'], emitter_kind='eleven-feature literal hinge interpolation',
                  selected_units=program.tolist())
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'hinge_output.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
actual = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(actual)
mask[:, program] = False
nrepair = int((actual & mask).sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output_and_cancellation': int(actual.sum()) - nrepair}
actual = gain_parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'native_text_repair': int(actual[ordinary].sum()), 'arithmetic_writer_gains': int(actual[pool].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/analyze_a84_hinge_digit_emitter_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', original_path, np, fp, wp, gp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
