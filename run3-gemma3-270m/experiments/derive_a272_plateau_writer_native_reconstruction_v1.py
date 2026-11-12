"""Allocate literal plateau hinges and reconstruct displaced native behavior.

Unit selection uses original HOST conditional native-feature energy. Each
smaller output pool is rebuilt independently from original HOST native
targets with the immutable direct reconstruction helper. New nonlinear
pieces are explicit hand hinges, evaluated only as ReLU ideals with error
bounds; all unchanged native features are official meter observations.
"""
import json
import math
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
install = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
gp = ROOT / 'artifacts/a271v1_plateau_literal_output_geometry.pt'
bp = ROOT / 'artifacts/a122v1_final_basis.pt'
bcp = ROOT / 'experiments/a122v1_native17/delta.pt'
np = ROOT / 'artifacts/a122v1_native8_word.pt'
ncp = ROOT / 'experiments/a122v1_native8/delta.pt'
hp = ROOT / 'artifacts/a83v1_host_top.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
word_reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
helper = ROOT / 'experiments/final_native_reconstruction_v1.py'
ch, ref, ref8 = [expand(h, p) for p in (base, bcp, ncp)]
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
geometry, b, n8, host_top, word, top, fmt = [torch.load(p, weights_only=True, mmap=True)
    for p in (gp, bp, np, hp, wp, tp, fp)]
for path, digest in geometry['report']['sources'].items(): assert sha(ROOT / path) == digest
for path, digest in geometry['report']['inputs'].items(): assert sha(ROOT / path) == digest
assert b['checkpoint_sha256'] == sha(bcp) and n8['checkpoint_sha256'] == sha(ncp)
assert word['checkpoint_sha256'] == sha(word_reference) and fmt['checkpoint_sha256'] == sha(format_reference)
assert torch.equal(b['inputs'][:384], host_top['inputs'][:384])
assert torch.equal(n8['inputs'][:384], b['inputs'][:384])
assert torch.equal(word['inputs'], top['inputs'])
assert torch.equal(word['inputs'][:256], b['inputs'][:256])
pre = 'model.layers.17.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = c['units'].long(); oldpool = c['output_pool'].long()
assert len(old_units) == 11 and len(oldpool) == 96
u8 = torch.tensor(c['negative_one_selector_a122']['programme_units'])
native8 = torch.arange(2048)[~torch.isin(torch.arange(2048), u8)]
allowed8 = {'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(8))):
        assert torch.equal(ch.get(name, h[name]), ref8.get(name, h[name])), name
    if name in (gn, un, dn, pn) or name in allowed8: continue
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
for suffix, rows in (('mlp.gate_proj.weight', native8), ('mlp.down_proj.weight', native8)):
    name = 'model.layers.8.' + suffix
    a, z = ch.get(name, h[name]), ref.get(name, h[name])
    if 'down_proj' in name: assert torch.equal(a[:, rows], z[:, rows]), name
    else: assert torch.equal(a[rows], z[rows]), name
assert b['captures']['word.8.program_features'].abs().max() == 0
gate8 = n8['captures']['8.pre'][:384].double().reshape(-1, 640) @ ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T
assert gate8.max() < -16
del gate8, ref8, n8, ref
for rp in (word_reference, format_reference):
    ref = expand(h, rp)
    for name in h:
        if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
installed = expand(h, install)
for name in (gn, un, dn, pn): assert torch.equal(ch[name], installed[name]), name
del installed
old_native = torch.arange(2048)[~torch.isin(torch.arange(2048), old_units)]
assert torch.equal(ch[gn][old_native], h[gn][old_native]) and torch.equal(ch[un][old_native], h[un][old_native])
assert word['captures']['17.features'][:, :, old_units].abs().max() == 0

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

old_input = b['captures']['word.17.pre'].double().reshape(-1, 640)
pgain = 1 + h[pre + 'pre_feedforward_layernorm.weight']
word_input = (fnorm(word['captures']['17.raw']) * pgain).double().reshape(-1, 640)
format_input = (fnorm(fmt['captures']['17.raw']) * pgain).double()
P = format_input[:, 8:13].reshape(-1, 640)
N = torch.cat([old_input, word_input, format_input[:, :8].reshape(-1, 640)])
assert (N @ ch[gn][old_units].double().T).max() < -16
model = NativeFinalReconstruction(b['captures']['word.17.features'], h[dn], 1 + h[pn].double(), training_tokens=256 * 64)
native_baseline = model.quality(ch[dn], 1 + ch[pn].double(), old_units)
donors = c['native_donor_repairs']['17']['donors'].long()
assert len(donors) == 512 and not torch.isin(donors, old_units).any()
# Conditional lost-feature energy, from the original HOST only. The one
# donor solve is a ranking statistic, not a fitted candidate residual.
X = model.FH[:model.nt] / model.hr[:model.nt]
D = X[:, donors]; sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
C = Z.T @ Z / len(Z); rhs = Z.T @ X / len(Z)
regularized = C.clone(); regularized.diagonal().add_(.001)
beta = torch.linalg.solve(regularized, rhs)
conditional = (X.square().mean(0) - 2 * (beta * rhs).sum(0) + (beta * (C @ beta)).sum(0)).clamp_min(0)
importance = (h[dn].double() * (1 + h[pn].double())[:, None]).square().sum(0)
priority = conditional * importance
priority[torch.cat([old_units, donors])] = torch.inf
extra = priority.argsort()[:17]
assert len(extra.unique()) == 17 and not torch.isin(extra, torch.cat([old_units, donors])).any()
hinge_units = torch.cat([old_units[2:5], extra])
units = torch.cat([old_units[:2], hinge_units, old_units[5:]])
assert len(units) == 28 and len(units.unique()) == 28
unit_report = {'extra_units': extra.tolist(), 'hinge_units': hinge_units.tolist(), 'programme_units': units.tolist(),
    'fixed_donor_count': 512, 'fixed_conditional_ridge': .001,
    'selected_conditional_costs': priority[extra].tolist(), 'conditional_variances': conditional[extra].tolist()}
del X, D, Z, C, rhs, regularized, beta, priority, conditional, importance
common = ch[gn][old_units[0]].double()
flag = torch.zeros(640, dtype=torch.float64); flag[476] = 1 / pgain[476].double()
digit = torch.zeros_like(flag)
digit[391] = c['piece']['source_flag_scale'] / c['piece']['source_digit_scale'] / pgain[391].double()
alpha = (flag * ch[un][old_units[0]].double()).sum() / flag.square().sum()
assert (ch[un][old_units[0]].double() - alpha * flag).abs().max() < 1e-6
assert (ch[un][old_units[1]].double() - alpha * digit).abs().max() < 1e-6
off = c['previous_hinge_emitter']['common_scale'] * flag - common
pf, po, no = P @ flag, P @ off, N @ off
assert pf.min() > 0 and no.min() > 0 and (P @ common).min() > 16
thresholds = torch.tensor(geometry['report']['literal_thresholds'], dtype=torch.float64)
assert len(thresholds) == 20
fields = digit[None] - thresholds[:, None] * flag
negative_fields = N @ fields.T
sharpnesses = [16., 64., 256., 1024., 4096.]
gate_trials = []; gate_choices = []
for sharp in sharpnesses:
    multipliers = ((sharp * negative_fields + 512) / no[:, None]).amax(0).clamp_min(0) * (1 + 1e-6)
    exact = sharp * fields - multipliers[:, None] * off
    gates = exact.float(); up = (alpha / sharp * common).float()
    rounding = ((gates.double() - exact) * pgain.double()).norm(dim=1) * 640**.5
    bound = (po.abs()[:, None] * multipliers + .25 + rounding) / (sharp * pf[:, None])
    row = {'sharpness': sharp, 'off_multipliers': multipliers.tolist(),
        'maximum_negative_gate': float((N @ gates.double().T).max()),
        'maximum_literal_hinge_error_bound': float(bound.max()),
        'maximum_gate_weight': float(gates.abs().max()), 'maximum_up_weight': float(up.abs().max())}
    row['eligible'] = bool(row['maximum_negative_gate'] < -500 and bound.max() < .002
        and max(row['maximum_gate_weight'], row['maximum_up_weight']) <= 1e6)
    gate_trials.append(row)
    if row['eligible']: gate_choices.append((row['maximum_literal_hinge_error_bound'], gates, up, row))
    print(json.dumps({'hinge_gates': row}), flush=True)
assert gate_choices, 'No declared sharpness yields quiet accurate hand hinges'
_, gates, up, gate_selection = min(gate_choices, key=lambda z: z[0])
G, U = ch[gn].clone(), ch[un].clone()
G[hinge_units] = gates; U[hinge_units] = up
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert torch.equal(G[native], h[gn][native]) and torch.equal(U[native], h[un][native])
all_word_gate = float((N @ G[units].double().T).max()); assert all_word_gate < -16
SF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048).clone()
SB = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
expected_format = fmt['topk_ids'][:, :, 0].reshape(-1).long()
format_gate = P @ G[hinge_units].double().T
format_up = P @ U[hinge_units].double().T
SF[:, hinge_units] = format_gate.clamp_min(0) * format_up
# An analytic tanh-GELU tail bound for these explicitly hand-built gates.
# No activation is evaluated; .25 is also a global uniform error bound.
absolute = format_gate.abs()
exponent = -2 * math.sqrt(2 / math.pi) * (absolute + .044715 * absolute.pow(3))
activation_error = torch.minimum(torch.full_like(absolute, .25), absolute * exponent.clamp_min(-700).exp())
feature_error = activation_error * format_up.abs()
hand_report = {'minimum_absolute_format_hinge_gate': float(absolute.min()),
    'maximum_uniform_GELU_feature_error_bound': float((.25 * format_up.abs()).max()),
    'maximum_format_tanh_GELU_feature_error_bound': float(feature_error.max()),
    'all_observed_word_programme_gate_maximum': all_word_gate,
    'scope': 'Real-arithmetic hand-hinge tail bound at reconstructed inputs and rounded weights; not a complete float32 model certificate'}
bm = json.loads(install.with_suffix('.manifest.json').read_text())
def changed_count(W, parameter):
    total = bm['changed_scalars']
    for name, value in ((gn, G), (un, U), (dn, W), (pn, parameter)):
        total += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
    return total

native_trials = []; candidates = []
for count, geom in sorted(geometry['geometries'].items(), key=lambda z: int(z[0])):
    pool = geom['pool'].long()
    assert geom['report']['eligible_geometry'] and count in ('64', '80')
    assert torch.equal(geom['thresholds'], thresholds)
    programme = torch.cat([geom['constant_down'][:, None], geom['cleanup_down'][:, :1],
        geom['hinge_down'], geom['cleanup_down'][:, 1:]], 1).float()
    assert programme.shape == (640, 28) and programme.abs().max() <= 1e6
    W, parameter, repair = model.reconstruct(units, programme, pool, donors)
    q = model.quality(W, 1 + parameter.double(), units)
    total = changed_count(W, parameter)
    row = {'pool_count': int(count), **q, 'native_reconstruction': repair,
        'total_changed_scalars': total, 'maximum_modified_weight': max(float(z.abs().max()) for z in (G, U, W, parameter))}
    row['eligible_native'] = bool(q['validation_native_branch_rms'] < .98 * native_baseline['validation_native_branch_rms']
        and total <= 5016268 and row['maximum_modified_weight'] <= 1e6)
    native_trials.append(row)
    if row['eligible_native']: candidates.append((W, parameter, geom, row))
    print(json.dumps({'native_reconstruction': row}), flush=True)

E = h['model.embed_tokens.weight']; final_gain = 1 + h['model.norm.weight'].double()
def readout(states, expected):
    x = (norm(states) * final_gain).float()
    maxima = torch.full((len(x),), -torch.inf); winners = torch.full_like(expected, -1)
    competitors = torch.full_like(maxima, -torch.inf); desired = (x * E[expected]).sum(1)
    for start in range(0, len(E), 2048):
        logits = x @ E[start:start + 2048].T
        value, index = logits.max(1); take = value > maxima
        maxima[take] = value[take]; winners[take] = start + index[take]
        local = expected - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        competitors = torch.maximum(competitors, logits.amax(1))
    return {'tokens': len(x), 'matches': int((winners == expected).sum()), 'minimum_target_margin': float((desired - competitors).min())}

oldF = b['captures']['word.17.features'][256:384].double().reshape(-1, 2048).clone()
oldF[:, old_units] = 0
oldB = b['captures']['word.17.raw'][256:384].double().reshape(-1, 640)
old_expected = host_top['topk_ids'][256:384, :, 0].reshape(-1).long()
freshF = word['captures']['17.features'][384:].double().reshape(-1, 2048).clone()
freshB = word['captures']['17.raw'][384:].double().reshape(-1, 640)
fresh_expected = top['topk_ids'][384:, :, 0].reshape(-1).long()
validation = []; best = None; original_scores = None
if candidates:
    original_scores = {'old': readout(oldB + norm(oldF @ ch[dn].double().T) * (1 + ch[pn].double()), old_expected),
        'fresh': readout(freshB + norm(freshF @ ch[dn].double().T) * (1 + ch[pn].double()), fresh_expected)}
    assert original_scores['fresh']['matches'] == 8003
    oldF[:, units] = 0; freshF[:, units] = 0
    for W, parameter, geom, row in candidates:
        gain = 1 + parameter.double()
        old_score = readout(oldB + norm(oldF @ W.double().T) * gain, old_expected)
        fresh_score = readout(freshB + norm(freshF @ W.double().T) * gain, fresh_expected)
        extra_old = old_score['matches'] - original_scores['old']['matches']
        extra_fresh = fresh_score['matches'] - original_scores['fresh']['matches']
        record = {**row, 'old_validation': old_score, 'fresh_validation': fresh_score,
            'extra_old_matches': extra_old, 'extra_fresh_matches': extra_fresh, 'format_validation': None, 'eligible': False}
        if extra_old >= 0 and extra_fresh >= 8 and extra_old + extra_fresh >= 16:
            format_score = readout(SB + norm(SF @ W.double().T) * gain, expected_format)
            raw_error_bound = feature_error @ W[:, hinge_units].double().norm(dim=0)
            record.update(format_validation=format_score, maximum_format_hand_hinge_raw_error_bound=float(raw_error_bound.max()))
            record['eligible'] = bool(format_score['matches'] == 2560 and format_score['minimum_target_margin'] > 1)
        validation.append(record)
        key = (extra_old + extra_fresh, extra_fresh, -row['validation_native_branch_rms'])
        if record['eligible'] and (best is None or key > best[0]): best = key, W, parameter, geom, record
        print(json.dumps({'final_validation': record}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'install_base_checkpoint': str(install.relative_to(ROOT)), 'selected': None if best is None else best[-1],
    'unit_allocation': unit_report, 'native_baseline': native_baseline,
    'native_training_windows': 256, 'native_validation_windows': 128, 'fresh_validation_windows': 128,
    'sharpness_trials': sharpnesses, 'gate_trials': gate_trials, 'selected_hinge_gates': gate_selection,
    'hand_hinge_bounds': hand_report, 'native_donor_ridge': .001,
    'native_strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
    'native_trials': native_trials, 'original_full_vocabulary_validation': original_scores,
    'full_vocabulary_validation': validation,
    'selection': 'Native branch RMS improves>=2percent; old full-vocabulary matches do not fall; fresh gains>=8 and old+fresh gains>=16; all2560format winners withmargin>1; caps',
    'unit_selection': 'Fixed17 lowest original HOST conditional-feature-variance times postgain-weighted Down energy; exclude old11 and existing512donors',
    'native_target': 'Original HOST local normalized MLP17 output on original native features; immutable one-shot donor/ordinary-gain/strong-row construction, never a candidate residual',
    'selector8_word_provenance': 'Native Gate rows and native Down columns bit-identical; only zero-on-word programme differences; upstream0..7 identity and original/new Gate zero checks',
    'scope': 'Exact old/fresh local input provenance with new hand-hinge ideals; installation on A259 changes the upstream trajectory and needs an official full score',
    'calibration_reuse': 'Old/fresh labels distinguish this solve split only. These calibration windows were used in earlier analyses; only tools/score.py evaluates untouched heldout text.',
    'no_model_execution_native_activation_replay_gradients_answers_or_iterative_residual_refits': True,
    'sources': {**geometry['report']['sources'], str(Path(__file__).relative_to(ROOT)): sha(__file__),
        str(helper.relative_to(ROOT)): sha(helper)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, install, gp, bp, bcp, np, ncp, hp, wp, tp, fp,
        word_reference, format_reference, base.parent / 'circuit.pt', install.with_suffix('.manifest.json'))},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a272v1_plateau_writer_native_reconstruction.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'gate': G, 'up': U, 'down': best[1], 'gain_parameter': best[2],
        'geometry': best[3], 'programme_units': units, 'hinge_units': hinge_units, 'donors': donors, 'report': report})
print(json.dumps({'native_baseline': native_baseline, 'selected': report['selected'],
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
