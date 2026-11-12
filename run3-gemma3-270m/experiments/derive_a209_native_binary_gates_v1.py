"""Whole-format gated quantizers for A205's independent native source bits.

Compare a literal paired-hinge decoder (26 GLUs) with a quadratic plateau
decoder (14 GLUs). Only individual source-digit labels and format positions
are used. A closed-form OFF field is exactly null on declared training equals
states and negative on text/other format positions. All nonlinear arithmetic
below is the explicitly hand-designed programme ReLU ideal, never a native
activation evaluation. This writes programme rows only, not a checkpoint.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'; ch = expand(h, base)
fp = ROOT / 'artifacts/a205v1_native_binary_source_transport.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
field, mix, w, e = [torch.load(p, weights_only=True, mmap=True) for p in (fp, mp, wp, ep)]
assert field['report']['selected']['eligible']
source_base = ROOT / field['report']['base_checkpoint']; reference = expand(h, source_base)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name.startswith('model.layers.3.self_attn.'):
        if name == 'model.layers.3.self_attn.v_proj.weight':
            assert torch.equal(ch.get(name, h[name]), h[name])
        else: assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
assert mix['report']['inputs'][str(source_base.relative_to(ROOT))] == sha(source_base)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'], mix['inputs'])
pre = 'model.layers.3.'
V = h[pre + 'self_attn.v_proj.weight'].double()
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
rows = field['input_rows'].double(); assert rows.shape == (13, 640)
C = rows[12]; B = rows[:12]
stage = torch.load(base.parent / 'circuit.pt', weights_only=True)['native_a_stage']
old_units = stage['units'].long(); assert len(old_units) == 4
Wprogramme = ch[pre + 'mlp.down_proj.weight'][:, old_units].double()
old_gate_rows = ch[pre + 'mlp.gate_proj.weight'][old_units]
assert torch.equal(old_gate_rows, old_gate_rows[:1].expand_as(old_gate_rows))
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
current = {}
for tag in ('word', 'format'):
    before = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    assert torch.equal(fnorm(before) * (1 + h[pre + 'input_layernorm.weight']), mix['records'][tag]['input'])
    raw = torch.einsum('nhsv,ohv->nso', mix['records'][tag]['mixtures'].double() @ V.T, O)
    current[tag] = (norm(before.double() + norm(raw) * gamma) * gp).float().double()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
bits = 2 * ((torch.arange(10)[:, None] >> torch.arange(4)[None]) & 1).double() - 1
positions = [3, 2, 1, 0]
def labels(tokens):
    match = tokens[..., None] == ids; assert match.sum(-1).eq(1).all()
    return match.long().argmax(-1)
current_labels = labels(w['inputs'][256:, positions[:3]])
historical, historical_labels = [], []
for path in check_paths:
    cap = torch.load(path, weights_only=True, mmap=True); sx = cap['captures']['3.input'].double()
    raw = sum(sx[:, pos] @ (O[:, head] @ V).T for head, pos in enumerate(positions))
    state = cap['captures']['2.out'][:, 8].double() + norm(raw) * gamma
    historical.append((norm(state) * gp).float().double())
    historical_labels.append(labels(cap['inputs'][:, positions[:3]]))

P = torch.cat([current['format'][:96, 8], historical[0][:32], historical[1][:32]])
_, singular, right = torch.linalg.svd(P / P.norm(dim=1, keepdim=True), full_matrices=True)
numerical_rank = int((singular > singular[0] * 1e-10).sum())
# Retain every row-space direction, including tiny singular directions.
# Dropping a small normalized singular value need not satisfy the absolute
# zero-constraint tolerance when the original rows have different norms.
rank = len(P); null = right[rank:].T
null_error = float((P @ null).abs().max())
assert null.shape[1] > 0 and null_error < 1e-8, (rank, numerical_rank, null_error)
non_equals = torch.tensor([j for j in range(14) if j != 8])
word_train = current['word'][:192].reshape(-1, 640)
format_train = current['format'][:96, non_equals].reshape(-1, 640)
negative_train = torch.cat([word_train, format_train])
negative_all = torch.cat([current['word'].reshape(-1, 640), current['format'][:, non_equals].reshape(-1, 640)])
XW, XN = word_train @ null, format_train @ null
CW, CN = XW.T @ XW / len(XW), XN.T @ XN / len(XN)
mw, mn = XW.mean(0), XN.mean(0)

# Positive bit intervals use the training groups only. No sum is computed.
interval_X = torch.cat([current['format'][:96, 8], historical[0], historical[1]])
interval_y = torch.cat([current_labels[:96], historical_labels[0], historical_labels[1]])
interval_codes = interval_X @ rows.T
ratios = interval_codes[:, :12] / interval_codes[:, 12:]
true_bits = bits[interval_y].reshape(-1, 12)
low = torch.stack([ratios[true_bits[:, j] > 0, j].min() for j in range(12)])
high = torch.stack([ratios[true_bits[:, j] > 0, j].max() for j in range(12)])
mid, radius = (low + high) / 2, (high - low) / 2
quadratic_scale = 1 / (mid.square() - radius.square() / 2)
assert low.min() > 0 and quadratic_scale.min() > 0
delta = .25
designs = {}
for label in ('paired_hinge_26', 'quadratic_plateau_14'):
    bases, ups = [(1 + delta) * C, (1 - delta) * C], [C, C]
    width = 26 if label == 'paired_hinge_26' else 14
    transform = torch.zeros(4, width, dtype=torch.float64)
    transform[0, 0] = 1 / (2 * delta); transform[0, 1] = -1 / (2 * delta)
    transform[1:] = -transform[:1]
    for j in range(12):
        head, bit = divmod(j, 4)
        if label == 'paired_hinge_26':
            bases.extend([B[j] + delta * C, B[j] - delta * C]); ups.extend([C, C])
            transform[head + 1, 2 + 2 * j] = 2 ** bit / (4.5 * 2 * delta)
            transform[head + 1, 3 + 2 * j] = -2 ** bit / (4.5 * 2 * delta)
        else:
            bases.append(B[j]); ups.append(2 * mid[j] * C - B[j])
            transform[head + 1, 2 + j] = 2 ** bit * quadratic_scale[j] / 4.5
    base_rows, up_rows = torch.stack(bases), torch.stack(ups)
    expected = bits[current_labels].reshape(-1, 12)
    if label == 'paired_hinge_26': expected = expected.repeat_interleave(2, 1)
    expected = torch.cat([torch.ones(len(expected), 2), expected], 1)
    designs[label] = {'base_gate_rows': base_rows, 'base_up_rows': up_rows,
        'transform': transform, 'expected_gate_signs': expected,
        'negative_training_base': negative_train @ base_rows.T}

negative_masses = [.01, .1, 1., 10., 100.]
ridges = [1e-6, 1e-4, .01, 1., 100.]
up_gain = 1e6
trials, selected = [], {}
for mass in negative_masses:
    covariance = (CW + mass * CN) / (1 + mass)
    rhs = -(mw + mass * mn) / (1 + mass)
    sd = covariance.diagonal().sqrt().clamp_min(1e-12)
    normalized = covariance / sd[:, None] / sd[None]
    ev, vec = torch.linalg.eigh((normalized + normalized.T) / 2); ev = ev.clamp_min(0)
    transformed = vec.T @ (rhs / sd)
    for ridge in ridges:
        off = null @ ((vec @ (transformed / (ev + ridge))) / sd)
        off_train = negative_train @ off
        off_max = float(off_train.max())
        for label, design in designs.items():
            rec = {'design': label, 'negative_format_mass': mass, 'OFF_ridge': ridge,
                'OFF_training_maximum': off_max, 'programme_units': len(design['base_gate_rows']), 'eligible': False}
            if off_max >= -1e-8:
                rec['reason'] = 'OFF field does not separate every training negative'; trials.append(rec); continue
            bound = (design['negative_training_base'] / (-off_train[:, None])).max().clamp_min(0)
            off_scale = 2 * float(bound) + 1e-6
            unscaled = design['base_gate_rows'] + off_scale * off
            train_eq = current['format'][:96, 8] @ unscaled.T
            expected = design['expected_gate_signs'][:96]
            positive = train_eq[expected > 0]
            negative = torch.cat([-train_eq[expected < 0], -(negative_train @ unscaled.T).flatten()])
            rec.update(OFF_scale=off_scale, minimum_unscaled_training_positive=float(positive.min()),
                       minimum_unscaled_training_negative_magnitude=float(negative.min()))
            if min(float(positive.min()), float(negative.min())) <= 0:
                rec['reason'] = 'Hand-programme gate signs fail before scaling'; trials.append(rec); continue
            gate_gain = 1.001 * max(64 / float(positive.min()), 32 / float(negative.min()))
            G = (gate_gain * unscaled).float()
            unscaled_down = Wprogramme @ design['transform']
            down_scale = min(1., 1e6 / float(unscaled_down.abs().max()))
            transform = design['transform'] * down_scale
            U = ((up_gain / down_scale) * design['base_up_rows']).float()
            down = (Wprogramme @ transform).float()
            rec.update(gate_gain=gate_gain, up_gain=up_gain / down_scale,
                       common_programme_down_scale=down_scale, maximum_gate_weight=float(G.abs().max()),
                       maximum_up_weight=float(U.abs().max()), maximum_programme_down_weight=float(down.abs().max()))
            if max(rec['maximum_gate_weight'], rec['maximum_up_weight'], rec['maximum_programme_down_weight']) > 1e6:
                rec['reason'] = 'A programme weight exceeds1e6'; trials.append(rec); continue
            word_gates = negative_all.float() @ G.T
            current_gates = current['format'][:, 8].float() @ G.T
            sign = design['expected_gate_signs']
            rec.update(maximum_all_negative_gate=float(word_gates.max()),
                minimum_current_positive_gate=float(current_gates[sign > 0].min()),
                maximum_current_negative_bit_gate=float(current_gates[sign < 0].max()),
                maximum_real_training_OFF_constraint_error=float((P @ off).abs().max()))
            diagnostics = []
            groups = [('current128', current['format'][:, 8], current_labels, True),
                      ('current_holdback32', current['format'][96:, 8], current_labels[96:], True)]
            groups += [(name, X, y, False) for name, X, y in zip(('historical_a', 'historical_stress', 'historical_c'), historical, historical_labels)]
            for name, X, y, factual in groups:
                gate = (X.float() @ G.T).double(); up = (X.float() @ U.T).double()
                # Only the constructed programme's ReLU saturation ideal.
                features = gate.clamp_min(0) * up
                effective = features @ transform.T
                valid_reference = effective[:, 0].gt(0) & effective.isfinite().all(1)
                got = 4.5 * (effective[valid_reference, 1:] / effective[valid_reference, :1]) + 4.5
                error = (got - y[valid_reference]).abs()
                finite_decoding = bool(valid_reference.all() and got.isfinite().all())
                expected_bits = bits[y].reshape(-1, 12)
                if label == 'paired_hinge_26': expected_bits = expected_bits.repeat_interleave(2, 1)
                expected_signs = torch.cat([torch.ones(len(X), 2), expected_bits], 1)
                signed = gate * expected_signs
                raw = features @ down.double().T
                reference_raw = effective @ Wprogramme.T
                diagnostics.append({'split': name, 'factual_current_mixtures': factual,
                    'finite_decoding': finite_decoding,
                    'invalid_reference_cases': int((~valid_reference).sum()),
                    'maximum_source_digit_error': float(error.max()) if finite_decoding else None,
                    'errors_per_place': error.amax(0).tolist() if finite_decoding else None,
                    'wrong_source_digits_with_valid_reference': int((got.round() != y[valid_reference]).sum()),
                    'minimum_effective_reference': float(effective[:, 0].min()),
                    'minimum_absolute_signed_gate': float(signed.min()),
                    'wrong_gate_signs': int((signed <= 0).sum()),
                    'rounded_programme_Down_relative_error': float((raw - reference_raw).norm() / reference_raw.norm()) if reference_raw.norm() > 0 else None})
            rec['source_diagnostics'] = diagnostics
            worst = max(x['maximum_source_digit_error'] if x['finite_decoding'] else float('inf') for x in diagnostics)
            rec['eligible'] = bool(G.isfinite().all() and U.isfinite().all() and down.isfinite().all()
                and rec['maximum_all_negative_gate'] < -16
                and rec['minimum_current_positive_gate'] > 32 and rec['maximum_current_negative_bit_gate'] < -16
                and all(x['finite_decoding'] and x['minimum_effective_reference'] > 0 and x['minimum_absolute_signed_gate'] > 16
                    and x['wrong_source_digits_with_valid_reference'] == 0 and x['wrong_gate_signs'] == 0 for x in diagnostics)
                and worst < .15)
            trials.append(rec)
            key = (worst, rec['maximum_gate_weight'])
            if rec['eligible'] and (label not in selected or key < selected[label][0]):
                selected[label] = key, G.clone(), U.clone(), down.clone(), transform.clone(), rec
    print(json.dumps({'negative_format_mass': mass, 'selected': {k: v[-1] for k, v in selected.items()}}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'negative_format_mass_trials': negative_masses, 'OFF_ridge_trials': ridges,
    'hinge_half_width': delta, 'constant_pair_midpoint': 1., 'fixed_common_up_gain': up_gain,
    'OFF_scale_rule': 'Twice the exact minimum training-negative sign-separation scale plus1e-6',
    'gate_gain_rule': '1.001 times the exact training bound for positive64 and negative32 margins',
    'programme_down_scale_rule': 'Uniform analytic scale<=1 to keep programme Down coefficients<=1e6, with its reciprocal on Up to preserve the programme raw amplitude',
    'training_equals_constraints': len(P), 'constraint_rank': rank, 'OFF_nullspace_dimension': null.shape[1],
    'numerical_training_equals_rank': numerical_rank, 'full_row_span_null_error': null_error,
    'training_word_windows': 192, 'validation_word_windows': 64, 'current_format_training_windows': 96, 'current_holdback_windows': 32,
    'historical_training_constraints_per_group': 32,
    'quadratic_positive_bit_interval_low': low.tolist(), 'quadratic_positive_bit_interval_high': high.tolist(),
    'quadratic_plateau_centers': mid.tolist(), 'quadratic_output_scales': quadratic_scale.tolist(),
    'quadratic_interval_target': 'Independent positive bit-code intervals only, never an operand-pair or sum',
    'historical_caveat': 'Old states with ideal routing; component diagnostics only, not exact current trajectories',
    'scope': 'Programme rows only; native unit allocation, displaced-feature repair and full scoring remain necessary',
    'only_hand_programme_ReLU_ideal_no_native_nonlinear_replay': True,
    'no_model_checkpoint_gradients_iterative_refit_or_answers': True,
    'trials': trials, 'selected': {k: v[-1] for k, v in selected.items()},
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, source_base, fp, mp, wp, ep] + check_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a209v1_native_binary_gates.json'; atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'designs': {k: {'gate_rows': v[1], 'up_rows': v[2], 'programme_down_columns': v[3],
        'effective_feature_transform': v[4], 'record': v[5]} for k, v in selected.items()}, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
