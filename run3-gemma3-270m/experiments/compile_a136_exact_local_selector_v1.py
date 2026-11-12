"""Compact local selector repair on the leading A123 exact input basis.

Original HOST native features at that input are the sole text target. A single
512-donor repair and five independent strong-row ridges replace the full
global selector fit. All programme coefficients and programme gains are fixed.
Recorded old-format native features plus absolute bounds for two restored
native units certify the format change without replaying native activations.
"""
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

h = host_state()
base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
cp = ROOT / 'experiments/a122v1_native8/delta.pt'
format_cp = ROOT / 'experiments/a120v1_rank_one_digit_value/delta.pt'
wp = ROOT / 'artifacts/a122v1_native8_word.pt'
fp = ROOT / 'artifacts/a122v1_format.pt'
hp = ROOT / 'artifacts/a73v1_host8_only.pt'
ch, native, old = [expand(h, p) for p in (base, cp, format_cp)]
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
w, fmt, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp, hp)]
assert w['checkpoint_sha256'] == sha(cp)
assert fmt['checkpoint_sha256'] == sha(format_cp)
assert torch.equal(w['inputs'], host['inputs'])
prefix = 'model.layers.8.'
gn, un, dn, pn = [prefix + s for s in (
    'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
    'post_feedforward_layernorm.weight')]
for name in h:
    upstream = any(name.startswith(f'model.layers.{i}.') for i in range(8))
    upstream |= name.startswith(prefix) and name not in (gn, un, dn, pn)
    if upstream:
        assert torch.equal(ch.get(name, h[name]), native.get(name, h[name])), name
        assert torch.equal(ch.get(name, h[name]), old.get(name, h[name])), name
for name in (gn, un, dn, pn):
    assert torch.equal(native.get(name, h[name]), h[name]), name
for key, value in w['captures'].items():
    if key.endswith('program_features') and int(key.split('.')[0]) < 8:
        assert value.abs().max() == 0, key
units = torch.tensor(c['negative_one_selector_a122']['programme_units'])
freed = torch.tensor(c['negative_one_selector_a122']['restored_native_units'])
programme = ch[dn][:, units].double()
gamma = 1 + ch[pn].double()
pool = (gamma.abs() > 1e5).nonzero().flatten()
assert len(units) == 9 and len(pool) == 2 and len(freed) == 2
fixed = programme.ne(0).any(1)
fixed[pool] = True
gate_max = float((w['captures']['8.pre'].double().reshape(-1, 640)
                  @ ch[gn][units].double().T).max())
assert gate_max < -16
donors = c['native_donor_repairs']['8']['donors'].long()
assert len(donors) == 512
model = NativeFinalReconstruction(w['captures']['8.features'], h[dn],
                                  1 + h[pn].double(), training_tokens=384 * 64)
baseline = model.quality(ch[dn], gamma, units)
W, parameter, repair = model.reconstruct(units, programme, pool, donors)
parameter = parameter.clone()
parameter[fixed] = ch[pn][fixed]
gain = 1 + parameter.double()
quality = model.quality(W, gain, units)
assert torch.equal(W[:, units], ch[dn][:, units])

PF = fmt['captures']['8.features'][:, 8:13].double().reshape(-1, 2048).clone()
PP = fmt['captures']['8.pre'][:, 8:13].double().reshape(-1, 640)
PF[:, freed] = 0
G, U = PP @ ch[gn][units].double().T, PP @ ch[un][units].double().T
PF[:, units] = G.clamp_min(0) * U
missing = (PP @ h[gn][freed].double().T).abs().add(.25)
missing *= (PP @ h[un][freed].double().T).abs()
raw = PF @ ch[dn].double().T
raw_error = .25 * U.abs() @ programme.abs().T
raw_error += missing @ ch[dn][:, freed].double().abs().T
lower = (raw.square().sum(1) + 640e-6).sqrt() - raw_error.norm(dim=1)
dW = W.double() - ch[dn].double()
difference = PF @ dW.T
delta_error = missing @ dW[:, freed].abs().T
d = difference.norm(dim=1) + delta_error.norm(dim=1)
assert (lower > d).all()
bound = 640**.5 * gain.abs() * (
    (difference.abs() + delta_error) / (lower - d)[:, None]
    + (raw.abs() + raw_error) * (d / (lower * (lower - d)))[:, None])
bound += 640**.5 * (gain - gamma).abs() * (raw.abs() + raw_error) / lower[:, None]
format_bound = float(bound.max())
F = model.FH[model.nt:].clone()
F[:, units] = 0
before = w['captures']['8.raw'][384:].double().reshape(-1, 640)
teacher = host['captures']['8.out'][384:].double().reshape(-1, 640)
teacher_rms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
state = {}
for name, down, g in (('previous', ch[dn], gamma), ('local_native', W, gain)):
    r = F @ down.double().T
    output = before + r / (r.square().mean(1, keepdim=True) + 1e-6).sqrt() * g
    state[name] = float(((output - teacher) / teacher_rms).square().mean().sqrt())
report = {
    'exact_upstream_provenance_verified': True, 'programme_count': 9,
    'donor_count': 512, 'donor_ridge': .001,
    'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
    'native_baseline': baseline, 'native_repair': repair,
    'quality_after_retaining_programme_gains': quality,
    'fixed_programme_gain_rows': fixed.nonzero().flatten().tolist(),
    'paired_original_HOST8_state_rms': state,
    'maximum_format_post_change_bound': format_bound,
    'maximum_word_programme_gate': gate_max,
    'format_bound': 'Recorded surviving native features; absolute bounds for two restored native units; bounded ReLU ideal only for hand-programme features.',
    'training_windows': 384, 'validation_windows': 128, 'format_windows': 512,
    'original_HOST_targets_no_answers_gradients_or_activation_replay': True}
out = ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'local_selector_repair.json', report)
print(json.dumps(report), flush=True)
assert quality['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
assert format_bound < 100
ch[dn], ch[pn] = W, parameter
c['exact_local_selector_repair_a136'] = report
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
changed = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(changed)
mask[:, units] = False
nr = int((changed & mask).sum())
categories[dn] = {'native_text_repair': nr, 'arithmetic_programme': int(changed.sum()) - nr}
changed = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'fixed_programme_gains': int(changed[fixed].sum()),
                  'native_text_repair': int(changed[~fixed].sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', cp, format_cp, wp, fp, hp],
     arguments={}, description=__doc__, extra=report, categories=categories)
