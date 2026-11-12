"""Combine nine-function cleanup with the independent 64-coordinate writer.

A154 supplies a conditional rank-four compression of the existing cleanup
operator. A145 supplies the protected-head five-function output geometry.
Their cancellation and emission pieces are combined by fixed weight algebra;
no whole-problem targets or previous native-fit residuals are used. Original
HOST local native reconstructions, vocabulary checks and native-format
uncertainty bounds are then applied independently to the combined geometry.
"""
import copy
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
bp = ROOT / 'artifacts/a122v1_final_basis.pt'
cp = ROOT / 'experiments/a122v1_native17/delta.pt'
np = ROOT / 'artifacts/a122v1_native8_word.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
hp = ROOT / 'artifacts/a83v1_host_top.pt'
ch, reference = expand(h, base), expand(h, cp)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
b, n8, stress, host_top = [torch.load(p, weights_only=True, mmap=True) for p in (bp, np, sp, hp)]
assert b['checkpoint_sha256'] == sha(cp)
assert stress['checkpoint_sha256'] == sha(ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt')
assert torch.equal(b['inputs'][:384], host_top['inputs'][:384])
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = c['units'].long(); pool = c['output_pool'].long()
gp = ROOT / 'artifacts/a154v1_conditional_cleanup_geometry.pt'
geometry = torch.load(gp, weights_only=True)
assert geometry['report']['base_sha256'] == sha(base)
assert len(old_units) == 11 and len(pool) == 96
oldpool = pool.clone()
for name in h:
    if name.startswith(prefix) and ('.mlp.' in name or name == pn):
        continue
    if name in {'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.down_proj.weight'}:
        continue
    assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
u8 = torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert b['captures']['word.8.program_features'].abs().max() == 0
assert (n8['captures']['8.pre'][:384].double().reshape(-1, 640) @ ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max() < -16
Pword = b['captures']['word.17.pre'].double().reshape(-1, 640)
Gprog, Uprog, Wprog = geometry['gate'], geometry['up'], geometry['down'].double()
count = len(Gprog); assert count == 9
pp = ROOT / 'artifacts/a145v1_literal_span_pool.pt'
writer = torch.load(pp, weights_only=True)['candidates']['64']
clean = Wprog.clone()
clean[oldpool, :5] -= c['piece']['output_gain'] * c['output_directions'].double()
clean[c['piece']['raw_anchor_coordinate']] = 0
clear = torch.tensor(c['piece']['cancellation_coordinates'])
mask = torch.ones(640, dtype=torch.bool); mask[clear] = False
cleanup_rounding = float(clean[mask].abs().max()); assert cleanup_rounding < .02
clean[mask] = 0
pool = writer['coordinates'].long()
Wprog = clean * writer['report']['cancellation_ratio']
Wprog[pool, :5] += writer['report']['raw_output_gain'] * writer['basis'].double()
Wprog[c['piece']['raw_anchor_coordinate'], 0] += writer['report']['raw_output_gain'] * writer['report']['anchor_amplitude']
Wprog = Wprog.float().double()
assert (Pword @ Gprog.double().T).max() < -32
donors = c['native_donor_repairs']['17']['donors'].long()
assert len(donors) == 512 and not torch.isin(old_units, donors).any()
available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
model = NativeFinalReconstruction(b['captures']['word.17.features'], h[dn], 1 + h[pn].double(), training_tokens=256 * 64)
baseline = model.quality(ch[dn], 1 + ch[pn].double(), old_units)
nt = model.nt
D = model.FH[:nt, donors] / model.hr[:nt]
P = model.FH[:nt, available] / model.hr[:nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
C = Z.T @ Z / len(Z); C.diagonal().add_(.001)
fit = torch.linalg.solve(C, Z.T @ P / len(Z)) / sd[:, None]
conditional = (P - D @ fit).square().mean(0)
post_cost = conditional * (model.WH[:, available] * model.gh[:, None]).square().sum(0)
raw_cost = conditional * model.WH[:, available].square().sum(0)
post_cost /= post_cost.sum(); raw_cost /= raw_cost.sum()
del D, P, Z, C, fit
trials = []; seen = set(); best = None
out = ROOT / 'experiments/a156v1_compact_cleanup_writer/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
for mass in [0., .01, .1, 1., 10.]:
    units = available[(post_cost + mass * raw_cost).argsort()[:count]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'raw_importance_mass': mass, 'duplicate_unit_set': True, 'units': units.tolist()})
        continue
    seen.add(key)
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    q = model.quality(W, 1 + parameter.double(), units)
    row = {'raw_importance_mass': mass, 'units': units.tolist(), 'old_units_retained': int(torch.isin(units, old_units).sum()),
           'native_repair': repair, **q}
    trials.append(row); print(json.dumps(row), flush=True)
    if best is None or q['validation_native_branch_rms'] < best[0]:
        best = q['validation_native_branch_rms'], W.clone(), parameter.clone(), units.clone(), row
    atomic_json(out.parent / 'allocation_trials.json', {'baseline': baseline, 'trials': trials})
report = {'baseline': baseline, 'available_native_units': len(available), 'programme_feature_count': count,
          'fixed_donor_count': 512, 'conditional_cost_ridge': .001, 'raw_importance_mass_trials': [0, .01, .1, 1, 10],
          'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'training_windows': 256, 'validation_windows': 128,
          'conditional_cleanup_geometry': geometry['report'], 'writer_geometry': writer['report'], 'discarded_cleanup_rounding': cleanup_rounding, 'arithmetic_operator_compressed_from_recorded_hand_field_covariance': True, 'old_units': old_units.tolist(),
          'trials': trials, 'selected': None if best is None else best[-1],
          'no_answers_gradients_or_native_activation_replay': True}
atomic_json(out.parent / 'compact_cleanup_writer.json', report)
assert best is not None and best[0] < .95 * baseline['validation_native_branch_rms'], 'Less than five-percent native branch improvement'
_, W, parameter, units, selected = best
G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Gprog, Uprog

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
def readout(states, wanted):
    x = (norm(states) * fg).float()
    desired = (x * E[wanted]).sum(1)
    other = torch.full_like(desired, -torch.inf)
    maximum = torch.full_like(desired, -torch.inf); winner = torch.full((len(x),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = x @ E[start:start + 4096].T
        value, index = logits.max(1); better = value > maximum
        maximum[better] = value[better]; winner[better] = start + index[better]
        local = wanted - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf; other = torch.maximum(other, logits.amax(1))
    return {'tokens': len(x), 'matches': int((winner == wanted).sum()), 'minimum_target_margin': float((desired - other).min())}, desired - other

before = b['captures']['word.17.raw'][256:].double().reshape(-1, 640)
FH = b['captures']['word.17.features'][256:].double().reshape(-1, 2048)
oldF = FH.clone(); oldF[:, old_units] = 0
newF = FH.clone(); newF[:, units] = 0
expected = host_top['topk_ids'][256:384, :, 0].reshape(-1).long()
oldscore, _ = readout(before + norm(oldF @ ch[dn].double().T) * (1 + ch[pn].double()), expected)
newscore, _ = readout(before + norm(newF @ W.double().T) * (1 + parameter.double()), expected)
report['full_vocabulary_validation'] = {'baseline': oldscore, 'candidate': newscore, 'required_extra_matches': 8}
atomic_json(out.parent / 'compact_cleanup_writer.json', report)
print(json.dumps({'full_vocabulary_validation': report['full_vocabulary_validation']}), flush=True)
assert newscore['matches'] >= oldscore['matches'] + 8, 'Less than eight additional calibration matches'

# Exact copied programme functions plus observed unchanged native features.
# The native functions restored in retired programme slots were not captured
# on these formats. Use only their uniform absolute bounds, not GELU replay.
SF = stress['captures']['17.features'][:, 8:13].double().reshape(-1, 2048).clone()
SB = stress['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
old_programme_features = SF[:, old_units].clone()
pre = norm(SB) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'].double())
common_factor = old_programme_features[:, 0] / (pre @ ch[un][old_units[0]].double())
new_programme_features = torch.cat([old_programme_features[:, :5], common_factor[:, None] * (pre @ Uprog[5:].double().T)], 1)
SF[:, old_units] = 0; SF[:, units] = new_programme_features
retired = old_units[~torch.isin(old_units, units)]
pre = norm(SB) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'].double())
native_gate = pre @ h[gn][retired].double().T
native_up = pre @ h[un][retired].double().T
feature_bound = (native_gate.abs() + .25) * native_up.abs()
raw_error = feature_bound @ W[:, retired].double().abs().T
R = SF @ W.double().T
gain = 1 + parameter.double()
raw_norm = (R.square().sum(1) + 640e-6).sqrt()
error_norm = raw_error.norm(dim=1)
lower = raw_norm - error_norm; assert lower.min() > 0
post_error = 640 ** .5 * gain.abs() * (raw_error / lower[:, None] + R.abs() * (error_norm / (raw_norm * lower))[:, None])
state = SB + norm(R) * gain
post_error_norm = post_error.norm(dim=1)
state_norm = (state.square().sum(1) + 640e-6).sqrt()
assert (state_norm > post_error_norm).all()
final_error_bound = 2 * 640 ** .5 * post_error_norm / (state_norm - post_error_norm)
maximum_readout_norm = 0.
for start in range(0, len(E), 4096):
    maximum_readout_norm = max(maximum_readout_norm, float((E[start:start + 4096].double() * fg).norm(dim=1).max()))
format_score, margin = readout(state, stress['topk_ids'][:, :, 0].reshape(-1).long())
certified_margin = margin.double() - 2 * maximum_readout_norm * final_error_bound
report['format_bound'] = {'restored_native_unit_count': len(retired), 'restored_native_units': retired.tolist(),
                          'zero-substitution_reference': format_score,
                          'minimum_certified_full_vocabulary_margin': float(certified_margin.min()),
                          'maximum_final_direction_error_bound': float(final_error_bound.max()),
                          'bound': 'abs(GELU(g)) <= abs(g)+.25, then two RMS perturbation inequalities; no native feature value replayed'}
report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out.parent / 'compact_cleanup_writer.json', report)
print(json.dumps({'format_bound': report['format_bound']}), flush=True)
assert certified_margin.min() > 1, 'Restored native-feature uncertainty consumes format margin'

ch[gn], ch[un], ch[dn], ch[pn] = G, U, W, parameter
c['units'] = units; c['piece']['selected_units'] = units.tolist()
c['compact_cleanup_writer_a156'] = report
c['output_pool'], c['output_directions'] = pool, writer['basis']
c['piece'].update(output_pool=pool.tolist(), output_pool_size=len(pool), raw_anchor_amplitude=writer['report']['anchor_amplitude'], output_gain=writer['report']['raw_output_gain'], cancellation_scale=writer['report']['cancellation_scale'])
c['piece']['cancellation_basis_rank'] = 4; c['piece']['programme_units'] = count
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
for name in (gn, un):
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
changed = W.view(torch.int32) != h[dn].view(torch.int32)
categories[dn] = {'arithmetic_programme': int(changed[:, units].sum()), 'native_text_repair': int(changed.sum() - changed[:, units].sum())}
changed = parameter.view(torch.int32) != h[pn].view(torch.int32)
mask = torch.zeros(640, dtype=torch.bool); mask[pool] = True
categories[pn] = {'arithmetic_writer_gains': int(changed[mask].sum()), 'native_text_repair': int(changed[~mask].sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py', ROOT / 'experiments/analyze_a154_conditional_cleanup_geometry_v1.py', ROOT / 'experiments/analyze_a145_literal_span_pool_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', bp, cp, np, sp, hp, gp, gp.with_suffix('.json'), pp, pp.with_suffix('.json')],
     arguments={}, description=__doc__, extra=report, categories=categories)
