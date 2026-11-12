"""Decode individual A digits using a native plus reference instead of BOS.

V3 and its existing single literal-digit direction remain unchanged. Three
per-source digit constraints and the sum of native source means define the
four-dimensional decoder. Unlabeled covariance and independent source-token
variation penalize nuisance; no whole-problem or answer regression occurs.
The separately screened A185 query routes the fourth head to plus.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
cp = base.parent / 'circuit.pt'; c = torch.load(cp, weights_only=True)
qp = ROOT / 'artifacts/a185v1_plus_reference_query.pt'
query = torch.load(qp, weights_only=True); assert query['report']['selected']['eligible']
vp = ROOT / 'artifacts/a120_rank_one_digit_value_v1.pt'
value = torch.load(vp, weights_only=True)
wp = ROOT / 'artifacts/a119v1_current_word.pt'
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
sp = [ROOT / f'artifacts/a39_inputs_{name}.pt' for name in ('syntax_a', 'syntax_stress', 'syntax_c')]
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
word = torch.load(wp, weights_only=True, mmap=True)
caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
checks = [torch.load(p, weights_only=True, mmap=True) for p in sp]
early, branches = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branches['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branches['inputs'])
XX = torch.cat([cap['captures']['3.input'] for cap in caps]).double()
L = value['input_digit_row'].double(); U = value['value_dual'].double()
prefix = 'model.layers.3.'
vn = prefix + 'self_attn.v_proj.weight'; un = prefix + 'mlp.up_proj.weight'
V = ch[vn].double(); assert torch.equal(ch[vn], value['V'])
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
pg = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
assert torch.equal(ch.get(prefix + 'self_attn.o_proj.weight', h[prefix + 'self_attn.o_proj.weight']), h[prefix + 'self_attn.o_proj.weight'])
assert V.shape == (256, 640) and U.shape == (256, 1) and L.shape == (640, 1)
assert (value['value_decoder'].double().T @ V - L.T).abs().max() < 1e-6
T = word['captures']['3.input'][:384].double().reshape(-1, 640)
CX = T.T @ T / len(T); del T
remainder = V - U @ L.T
maps = [gamma[:, None] * (O[:, head] @ V) for head in range(4)]
remaining_maps = [gamma[:, None] * (O[:, head] @ remainder) for head in range(4)]
raw_maps = [O[:, head] @ V for head in range(4)]
positions = [3, 2, 1, 4]
allowed = pg.abs() >= .05
allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
rows, targets, means, source_terms = [], [], [], []
within = torch.zeros(len(idx), len(idx), dtype=torch.float64)
for head, position in enumerate(positions):
    if head < 3:
        rows.append((gamma[:, None] * (O[:, head] @ U))[idx, 0])
        target = torch.zeros(4, dtype=torch.float64); target[head] = 1
        targets.append(target)
    # The reference head has no decoded digit channel. Include its entire
    # native value, including any projection onto the literal digit axis.
    mean_map = remaining_maps[head] if head < 3 else maps[head]
    y = XX[:, position] @ mean_map.T
    mean = y[:, idx].mean(0); means.append(mean)
    centered = y[:, idx] - mean
    within += centered.T @ centered / (4 * len(centered))
    desired = torch.zeros(len(XX), 4, dtype=torch.float64)
    if head < 3: desired[:, head] = (XX[:, position] @ L)[:, 0]
    source_terms.append((XX[:, position] @ maps[head].T, desired))
rows.append(sum(means)); targets.append(torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
M, target = torch.stack(rows), torch.stack(targets)
length = M.norm(dim=1); M /= length[:, None]; target /= length[:, None]
assert torch.linalg.matrix_rank(M) == 4
CW = sum(mapping[idx] @ CX @ mapping[idx].T for mapping in maps) / 4

def norm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state2 = early['captures']['format.2.raw'] + norm(branches['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
state3 = state2 + norm(branches['captures']['format.3.attention.raw']) * gamma.float()
assert torch.equal(state3, early['captures']['format.3.raw'])
fresh_input = norm(state2) * (1 + h[prefix + 'input_layernorm.weight'])
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
datasets = []
for label, cap in zip(('syntax_a', 'syntax_stress', 'syntax_c'), checks):
    datasets.append((label, cap['captures']['3.input'].double(), cap['captures']['2.out'][:, 8].double(), cap['inputs']))
datasets.append(('current128', fresh_input.double(), state2[:, 8].double(), early['inputs'][256:, :14]))
decoded, backgrounds = [], []
for label, sx, before, tokens in datasets:
    raw = sum(sx[:, position] @ raw_maps[head].T for head, position in enumerate(positions))
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    wanted = torch.stack([(tokens[:, position, None] == ids[None]).long().argmax(1) for position in (3, 2, 1)], 1)
    decoded.append((label, raw * gamma / rms, before, rms, wanted))
    if label in ('syntax_a', 'syntax_stress'): backgrounds.append(before * rms)
    if label == 'current128': backgrounds.append((before * rms)[:96])
background = torch.cat(backgrounds)[:, idx]
CB = background.T @ background / len(background)
within_mass = 10000.
backgrounds_to_try = [0., 1., 100., 10000., 1000000.]
gain_choices = [1e3, 1e4, 1e5, 3e5, 1e6]
trials, best = [], None
for background_mass in backgrounds_to_try:
    Q = CW + within_mass * within + background_mass * CB
    Q.diagonal().add_(1e-6 * CW.diagonal().mean())
    Z = torch.linalg.solve(Q, M.T)
    frame = torch.zeros(640, 4, dtype=torch.float64)
    frame[idx] = Z @ torch.linalg.solve(M @ Z, target)
    inverse_pg = torch.where(pg.abs() > 1e-9, 1 / pg, torch.zeros_like(pg))
    up = (frame[:, [3, 0, 1, 2]] * inverse_pg[:, None]).T
    possible = [g for g in gain_choices if float(up.abs().max()) * g <= 1e6]
    if not possible:
        trials.append({'background_mass': background_mass, 'eligible': False, 'reason': 'No allowed Up gain fits weight bound'})
        continue
    gain = max(possible)
    quantized_up = (gain * up).float()
    effective = (quantized_up.double().T * pg[:, None]) / gain
    effective = effective[:, [1, 2, 3, 0]]
    offsets = [mean @ effective[idx] for mean in means]
    errors = [y @ effective - desired - offsets[head] for head, (y, desired) in enumerate(source_terms)]
    source_bound = sum(float(error.abs().max()) for error in errors)
    splits = []
    for label, branch, before, rms, wanted in decoded:
        combined = (branch + before) @ effective
        got = 4.5 * combined[:, :3] / combined[:, 3:] + 4.5
        error = (got - wanted).abs()
        row = {'split': label, 'minimum_reference': float(combined[:, 3].min()),
            'maximum_individual_digit_errors': error.amax(0).tolist(),
            'wrong_individual_digits': int((got.round() != wanted).sum()),
            'background_bound_before_normalization': float(((before @ effective) * rms).abs().max())}
        if label == 'current128':
            row['validation32_maximum_individual_digit_errors'] = error[96:].amax(0).tolist()
            row['validation32_wrong_individual_digits'] = int((got[96:].round() != wanted[96:]).sum())
        splits.append(row)
    criterion = source_bound + max(row['background_bound_before_normalization'] for row in splits[:2])
    row = {'background_mass': background_mass, 'common_Up_gain': gain,
        'independent_source_error_bound': source_bound, 'selection_metric': criterion,
        'maximum_Up_weight': float(quantized_up.abs().max()),
        'maximum_readout_weight': float(effective.abs().max()), 'splits': splits}
    row['eligible'] = all(x['minimum_reference'] > 0 and x['wrong_individual_digits'] == 0
        and max(x['maximum_individual_digit_errors']) < (.2 if x['split'] == 'current128' else .3) for x in splits)
    trials.append(row); print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or criterion < best[0]):
        best = criterion, quantized_up, effective, row
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'source_positions': positions, 'constant_reference': 'Digit-head residual means plus the complete native plus-head mean at offset-4',
    'V_unchanged': True, 'reference_head_uses_complete_value_mean_and_covariance': True, 'background_mass_trials': backgrounds_to_try,
    'common_Up_gain_choices': gain_choices, 'within_source_mass': within_mass, 'metric_ridge_fraction': 1e-6,
    'source_token_windows': len(XX), 'format_background_training_windows': len(background),
    'historical_validation_windows': 512, 'current_validation_windows': 32,
    'query_geometry': query['report']['selected'], 'trials': trials,
    'selected': None if best is None else best[-1],
    'literal_ideal_routing_only': True, 'prefix_and_actual_arithmetic_not_yet_scored': True,
    'no_whole_problem_answer_regression_gradients_or_native_activation_replay': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, qp, vp, wp, ep, bp] + pp + sp},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a186v2_plus_reference_frame.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'up': best[1], 'frame': best[2], 'query': query['query'], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
