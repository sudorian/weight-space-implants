"""Merge two identical arithmetic features and restore one native unit.

Only recorded native features and linear/RMS algebra are used. Every count
starts independently from HOST; all arithmetic functions remain fixed.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from native_fixed_gain_reconstruction_v2 import NativeFixedGainReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
reference = ROOT / meta['base']; assert sha(reference) == meta['base_sha256']
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ch = expand(h, base); ref = expand(h, reference)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
bm = json.loads(base.with_suffix('.manifest.json').read_text())
nt = 192 * 64
ridges = [1e-6, 1e-5, 1e-4, .001, .01]
expected_donors = [64, 512, 512, 512, 128, 128, 512, 512]

def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

teacher_states = {}
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(8):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    teacher_states[layer] = state.reshape(-1, 640).clone()
del state


layer = 2; prefix = 'model.layers.2.'
gn, un, dn, pn = [prefix + suffix for suffix in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = torch.tensor(meta['native_restored_programme_units']['2'])
assert old_units.tolist() == [80, 342, 529, 727, 1634]
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.', 'model.layers.2.')):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert torch.equal(ch[gn][529], ch[gn][727]) and torch.equal(ch[un][529], ch[un][727])
assert not (ch[dn][:, 529].ne(0) & ch[dn][:, 727].ne(0)).any()
mask = (ch[dn].view(torch.int32) != h[dn].view(torch.int32)).clone(); mask[:, old_units] = False
pool = (mask.sum(1) > 1000).nonzero().flatten()
ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
donors = (mask[ordinary].sum(0) > int(ordinary.sum()) / 2).nonzero().flatten()
assert len(donors) == 512 and not torch.isin(old_units, donors).any()
FH = a['captures']['2.native_features'].reshape(-1, 2048).double()
before = a['captures']['2.raw'].reshape(-1, 640).double()
teacher = teacher_states[2].double(); trms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
gain = 1 + ch[pn].double()
model = NativeFixedGainReconstruction(FH, h[dn], 1 + h[pn].double(), ch[dn], gain, old_units, training_tokens=nt)
def quality(W, units):
    F = FH[nt:].clone(); F[:, units] = 0
    raw = F @ W.double().T; post = norm(raw) * gain; state = before[nt:] + post
    return {**model.quality(W, gain, units),
        'validation_paired_HOST_state_relative_rms': float(((state - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_relative_raw_RMS_change': float(((raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() / model.reference[nt:] - 1).abs().max())}
baseline = quality(ch[dn], old_units)
trials = []; best = None
for restored, retained in [(529, 727), (727, 529)]:
    units = old_units[old_units != restored]
    programme = ch[dn][:, units].clone()
    programme[:, (units == retained).nonzero().item()] += ch[dn][:, restored]
    G, U = ch[gn].clone(), ch[un].clone()
    G[restored] = h[gn][restored]; U[restored] = h[un][restored]
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
    assert torch.equal(G[native], h[gn][native]) and torch.equal(U[native], h[un][native])
    old_product = ch[dn][:, old_units].double() @ ch[un][old_units].double()
    new_product = programme.double() @ U[units].double()
    product_error = float((old_product - new_product).abs().max())
    assert product_error == 0
    W, parameter, repair = model.reconstruct(units, programme, pool, donors)
    assert torch.equal(parameter, ch[pn]) and torch.equal(W[:, units], programme)
    q = quality(W, units)
    tensors = {gn: G, un: U, dn: W}
    count = bm['changed_scalars']
    for name, value in tensors.items():
        count += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
    row = {'restored_native_unit': restored, 'retained_reference_unit': retained, 'programme_units': units.tolist(),
        'exact_double_programme_map_max_error': product_error, 'native_reconstruction': repair, **q,
        'changed_scalars': count, 'maximum_weight': max(float(v.abs().max()) for v in tensors.values())}
    row['eligible_local'] = bool(q['validation_native_branch_rms'] <= .95 * baseline['validation_native_branch_rms']
        and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
        and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
        and count <= 5016268 and row['maximum_weight'] <= 1e6 and all(t.isfinite().all() for t in tensors.values()))
    trials.append(row); print(json.dumps(row), flush=True)
    if row['eligible_local'] and (best is None or q['validation_native_branch_rms'] < best[0]): best = q['validation_native_branch_rms'], tensors, row
helper = ROOT / 'experiments/native_fixed_gain_reconstruction_v2.py'
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'merged_original_units': [529, 727], 'gate_and_up_bit_identical': True, 'programme_down_supports_disjoint': True,
    'native_ridge_trials': ridges, 'training_windows': 192, 'validation_windows': 64,
    'exact_upstream_and_layer2_parent_identity': True, 'trials': trials, 'selected': None if best is None else best[-1],
    'next_required': 'Restored native-feature format bound and separately declared compiler/fullscore',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), str(helper.relative_to(ROOT)): sha(helper),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, ap, lp, cp, hp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a313v1_duplicate_reference_compression.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'weights': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
