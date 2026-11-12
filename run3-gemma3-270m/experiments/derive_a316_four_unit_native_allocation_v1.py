"""Allocate an exact four-feature programme using original native statistics.

The original A259 normalizer is fixed for every original-HOST reconstruction.
No fitted residual, native activation replay or answer target is used.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from native_fixed_gain_reconstruction_v2 import NativeFixedGainReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
sp = ROOT / 'artifacts/a313v1_duplicate_reference_compression.pt'
meta, certificate = [json.loads(p.read_text()) for p in (lp, cp)]
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
reference = ROOT / meta['base']; assert sha(reference) == meta['base_sha256']
ch, ref = expand(h, base), expand(h, reference)
a, host, source = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, sp)]
assert torch.equal(a['inputs'], host['inputs'])
for path, digest in {**source['report']['inputs'], **source['report']['sources']}.items():
    assert sha(ROOT / path) == digest, path
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.', 'model.layers.2.')):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
bm = json.loads(base.with_suffix('.manifest.json').read_text())
nt = 192 * 64
masses = [0., .01, .1, 1., 10.]
ridges = [1e-6, 1e-5, 1e-4, .001, .01]
prefix = 'model.layers.2.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                    'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = torch.tensor([80, 342, 529, 727, 1634])
source_units = torch.tensor(source['report']['selected']['programme_units'])
assert source_units.tolist() == [80, 342, 529, 1634]
Gprog, Uprog = [source['weights'][name][source_units].clone() for name in (gn, un)]
Wprog = source['weights'][dn][:, source_units].clone()
mask = ch[dn].view(torch.int32) != h[dn].view(torch.int32); mask[:, old_units] = False
pool = (mask.sum(1) > 1000).nonzero().flatten()
ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
donors = (mask[ordinary].sum(0) > int(ordinary.sum()) / 2).nonzero().flatten()
assert len(donors) == 512 and not torch.isin(old_units, donors).any()
available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
FH = a['captures']['2.native_features'].reshape(-1, 2048).double()
before = a['captures']['2.raw'].reshape(-1, 640).double()
gain = 1 + ch[pn].double()
model = NativeFixedGainReconstruction(FH, h[dn], 1 + h[pn].double(), ch[dn], gain,
                                      old_units, training_tokens=nt)
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
teacher = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(3):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        teacher += fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
teacher = teacher.reshape(-1, 640).double()
trms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def quality(W, units):
    F = FH[nt:].clone(); F[:, units] = 0
    raw = F @ W.double().T; state = before[nt:] + norm(raw) * gain
    return {**model.quality(W, gain, units),
        'validation_paired_HOST_state_relative_rms': float(((state - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher[nt:])).square().mean().sqrt())}
baseline = quality(source['weights'][dn], source_units)
assert abs(baseline['validation_native_branch_rms'] - source['report']['selected']['validation_native_branch_rms']) < 1e-12

# Each unit cost is a statistic of the original HOST conditional feature.
D = FH[:nt, donors] / model.hr[:nt]
P = FH[:nt, available] / model.hr[:nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
C = Z.T @ Z / nt; C.diagonal().add_(.001)
fit = torch.linalg.solve(C, Z.T @ P / nt) / sd[:, None]
residual = P - D @ fit
RN = model.RH[:nt] / model.hr[:nt]
S = RN.square().sum(1, keepdim=True) + 640e-6 / model.hr[:nt].square()
weighted_energy = (RN.square() * model.gh.square()).sum(1, keepdim=True)
post_cost, raw_cost = [], []
for start in range(0, len(available), 128):
    columns = available[start:start + 128]
    W = model.WH[:, columns]; e = residual[:, start:start + 128]
    dot = RN @ W; weighted_dot = RN @ (model.gh.square()[:, None] * W)
    wn = W.square().sum(0); wg = (W.square() * model.gh.square()[:, None]).sum(0)
    Sn = (S - 2 * e * dot + e.square() * wn).clamp_min(1e-20)
    numerator = weighted_energy - 2 * e * weighted_dot + e.square() * wg
    cross = weighted_energy - e * weighted_dot
    cost = 640 * (numerator / Sn + weighted_energy / S - 2 * cross / (Sn * S).sqrt())
    post_cost.append(cost.clamp_min(0).mean(0)); raw_cost.append((e.square() * wn).mean(0))
post_cost, raw_cost = torch.cat(post_cost), torch.cat(raw_cost)
post_cost /= post_cost.sum(); raw_cost /= raw_cost.sum()
del D, P, sd, Z, C, fit, residual, RN, S, weighted_energy, W, e, dot, weighted_dot, Sn, numerator, cross, cost

trials = []; seen = set(); best = None
for mass in masses:
    units = available[(post_cost + mass * raw_cost).argsort()[:4]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'raw_energy_mass': mass, 'units': units.tolist(), 'duplicate': True}); continue
    seen.add(key)
    G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Gprog, Uprog
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    assert torch.equal(parameter, ch[pn]) and torch.equal(W[:, units], Wprog)
    assert torch.equal(G[units], Gprog) and torch.equal(U[units], Uprog)
    q = quality(W, units); tensors = {gn: G, un: U, dn: W}
    count = bm['changed_scalars']
    for name, value in tensors.items():
        count += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
    row = {'raw_energy_mass': mass, 'units': units.tolist(), 'native_reconstruction': repair,
           **q, 'changed_scalars': count, 'maximum_weight': max(float(t.abs().max()) for t in tensors.values())}
    row['eligible_local'] = bool(q['validation_native_branch_rms'] <= .95 * baseline['validation_native_branch_rms']
        and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
        and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
        and count <= 5016268 and row['maximum_weight'] <= 1e6 and all(t.isfinite().all() for t in tensors.values()))
    trials.append(row); print(json.dumps(row), flush=True)
    if row['eligible_local'] and (best is None or q['validation_native_branch_rms'] < best[0]):
        best = q['validation_native_branch_rms'], tensors, row
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline_A313': baseline,
    'original_normalizer': 'A259 with original five programme features zero, fixed for every trial',
    'original_programme_units': old_units.tolist(), 'source_programme_units': source_units.tolist(),
    'raw_energy_masses': masses, 'conditional_ridge': .001, 'native_ridges': ridges,
    'training_windows': 192, 'validation_windows': 64, 'trials': trials, 'selected': None if best is None else best[-1],
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/native_fixed_gain_reconstruction_v2.py': sha(ROOT / 'experiments/native_fixed_gain_reconstruction_v2.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, ap, lp, cp, hp, sp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a316v1_four_unit_native_allocation.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'weights': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
