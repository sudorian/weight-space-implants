"""Allocate unchanged arithmetic functions by conditional HOST-boundary cost.

A fixed donor projection measures each native feature's residual variance.
This version weights that residual by the original HOST winner/runner-up
readout at each calibration token and fixed HOST normalizers, rather than a
uniform hidden-coordinate norm. Five raw-energy masses define independent
unit allocations; every native reconstruction targets original HOST values.
No gradients, iterative unit changes, model replay or sum labels are used.
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
htp = ROOT / 'artifacts/a56v1_host.pt'
ch, reference = expand(h, base), expand(h, cp)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
b, n8, stress, host_top = [torch.load(p, weights_only=True, mmap=True) for p in (bp, np, sp, hp)]
assert b['checkpoint_sha256'] == sha(cp)
assert stress['checkpoint_sha256'] == sha(ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt')
assert torch.equal(b['inputs'][:384], host_top['inputs'][:384])
host_final = torch.load(htp, weights_only=True, mmap=True)
assert torch.equal(b['inputs'][:384],host_final['inputs'][:384])
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = c['units'].long(); pool = c['output_pool'].long()
assert len(old_units) == 11 and len(pool) == 96
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
Gprog = ch[gn][old_units].clone(); Uprog = ch[un][old_units].clone(); Wprog = ch[dn][:, old_units].double().clone()
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
residual_features = P-D@fit
conditional = residual_features.square().mean(0)
embedding = h['model.embed_tokens.weight']; final_gain = 1+h['model.norm.weight'].double()
host_ids = host_top['topk_ids'][:384].reshape(-1,8).long()
host_gaps = (host_top['topk_values'][:384].reshape(-1,8)[:,0]-host_top['topk_values'][:384].reshape(-1,8)[:,1]).double().clamp_min(0)
margin_weights = 1/(host_gaps+.1);margin_weights/=margin_weights[:nt].mean();margin_weights=margin_weights.clamp(max=100)
readout_directions = (embedding[host_ids[:,0]].double()-embedding[host_ids[:,1]].double())*final_gain
teacher = host_final['captures']['finalraw'][:384].double().reshape(-1,640)
teacher_rms = (teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
projected_coefficients = ((readout_directions[:nt]*model.gh)/teacher_rms[:nt]) @ model.WH[:,available]
post_cost = ((residual_features*projected_coefficients).square()*margin_weights[:nt,None]).mean(0)
validation_rivals = embedding[host_ids[nt:]].double()*final_gain[None,None]
validation_before = b['captures']['word.17.raw'][256:].double().reshape(-1,640)
def allocation_quality(W,parameter,units):
    features = model.FH[nt:].clone();features[:,units]=0
    raw = features@W.double().T
    state = validation_before+raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*(1+parameter.double())
    scores = torch.einsum('nd,nkd->nk',state,validation_rivals)
    normalized = state/(state.square().mean(1,keepdim=True)+1e-6).sqrt()
    difference = normalized-teacher[nt:]/teacher_rms[nt:]
    errors = (difference*readout_directions[nt:]).sum(1)
    return {'validation_HOST_top8_matches':int((scores.argmax(1)==0).sum()),
            'validation_boundary_score_rms':float((errors.square()*margin_weights[nt:]).sum().sqrt()/margin_weights[nt:].sum().sqrt())}
base_boundary = allocation_quality(ch[dn],ch[pn],old_units)
del residual_features,projected_coefficients
raw_cost = conditional * model.WH[:, available].square().sum(0)
post_cost /= post_cost.sum(); raw_cost /= raw_cost.sum()
del D, P, Z, C, fit
trials = []; seen = set(); best = None
out = ROOT / 'experiments/a164v1_boundary_emitter_allocation/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
for mass in [0., .01, .1, 1., 10.]:
    units = available[(post_cost + mass * raw_cost).argsort()[:11]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        trials.append({'raw_importance_mass': mass, 'duplicate_unit_set': True, 'units': units.tolist()})
        continue
    seen.add(key)
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    q = model.quality(W, 1 + parameter.double(), units)
    q.update(allocation_quality(W,parameter,units))
    row = {'raw_importance_mass': mass, 'units': units.tolist(), 'old_units_retained': int(torch.isin(units, old_units).sum()),
           'native_repair': repair, **q}
    trials.append(row); print(json.dumps(row), flush=True)
    key = (q['validation_HOST_top8_matches'],-q['validation_boundary_score_rms'])
    if q['validation_native_branch_rms']<1.25*baseline['validation_native_branch_rms'] and (best is None or key > best[0]):
        best = key, W.clone(), parameter.clone(), units.clone(), row
    atomic_json(out.parent / 'allocation_trials.json', {'baseline': baseline, 'trials': trials})
report = {'baseline': baseline, 'baseline_boundary':base_boundary, 'allocation_metric':'Fixed conditional native-feature residual times original HOST winner/runner-up readout coefficient, weighted by inverse HOST margin at fixed HOST normalizers', 'margin_floor':.1, 'maximum_token_weight':100, 'available_native_units': len(available), 'programme_feature_count': 11,
          'fixed_donor_count': 512, 'conditional_cost_ridge': .001, 'raw_importance_mass_trials': [0, .01, .1, 1, 10],
          'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'training_windows': 256, 'validation_windows': 128,
          'arithmetic_functions_copied_bit_exactly': True, 'old_units': old_units.tolist(),
          'trials': trials, 'selected': None if best is None else best[-1],
          'no_answers_gradients_or_native_activation_replay': True}
atomic_json(out.parent / 'boundary_emitter_allocation.json', report)
assert best is not None, 'No allocation preserves the native branch within25percent'
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
atomic_json(out.parent / 'boundary_emitter_allocation.json', report)
print(json.dumps({'full_vocabulary_validation': report['full_vocabulary_validation']}), flush=True)
assert newscore['matches'] >= oldscore['matches'] + 8, 'Less than eight additional calibration matches'

# Exact copied programme functions plus observed unchanged native features.
# The native functions restored in retired programme slots were not captured
# on these formats. Use only their uniform absolute bounds, not GELU replay.
SF = stress['captures']['17.features'][:, 8:13].double().reshape(-1, 2048).clone()
SB = stress['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
old_programme_features = SF[:, old_units].clone()
SF[:, old_units] = 0; SF[:, units] = old_programme_features
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
atomic_json(out.parent / 'boundary_emitter_allocation.json', report)
print(json.dumps({'format_bound': report['format_bound']}), flush=True)
assert certified_margin.min() > 1, 'Restored native-feature uncertainty consumes format margin'

ch[gn], ch[un], ch[dn], ch[pn] = G, U, W, parameter
c['units'] = units; c['piece']['selected_units'] = units.tolist()
c['boundary_emitter_allocation_a164'] = report
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
     sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', bp, cp, np, sp, hp, htp],
     arguments={}, description=__doc__, extra=report, categories=categories)
