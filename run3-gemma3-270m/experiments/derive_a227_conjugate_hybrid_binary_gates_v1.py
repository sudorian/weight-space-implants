"""Conjugate quadratic and paired-hinge hybrids for independent source bits.

Compare five hand programmes with14/17/20/23/26GLUs. The largest0/1/2/3/4bits use exact paired hinges; remaining bits use a conjugate quadratic plateau whose Up includes the same OFF field as its Gate. Only individual source-digit labels and format positions
are used. A closed-form OFF field minimizes format-relative output energy on1120
training equals states while reconstructing a negative field on both actual native-V ordinary-text states and the A174/A176 native-feature surrogate states. All nonlinear arithmetic
below is the explicitly hand-designed programme ReLU ideal, never a native
activation evaluation. This writes programme rows only, not a checkpoint.
"""
import argparse, json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

parser = argparse.ArgumentParser(); parser.add_argument('--field-key', required=True)
args = parser.parse_args()
h = host_state(); base = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'; ch = expand(h, base)
fp = ROOT / 'artifacts/a221v1_native_binary_field_catalog.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
catalog, mix, w, e = [torch.load(p, weights_only=True, mmap=True) for p in (fp, mp, wp, ep)]
entry = catalog['alternatives'][args.field_key]
field = {'input_rows': entry['input_rows'], 'report': dict(catalog['report'], selected=entry['record'])}
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
layout = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
native = torch.load(ap, weights_only=True, mmap=True)
assert torch.equal(native['inputs'], w['inputs'][:256])
surrogate_word = (fnorm(native['captures']['3.raw']) * (1 + h[pre + 'pre_feedforward_layernorm.weight'])).double()
assert surrogate_word.shape == current['word'].shape
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

P = torch.cat([current['format'][:96, 8], historical[0], historical[1]])
positive_reference = P @ C
assert positive_reference.min() > 0
relative_positive = P / positive_reference[:, None]
non_equals = torch.tensor([j for j in range(14) if j != 8])
actual_word_train = current['word'][:192].reshape(-1, 640)
surrogate_word_train = surrogate_word[:192].reshape(-1, 640)
word_train = torch.cat([actual_word_train, surrogate_word_train])
format_train = current['format'][:96, non_equals].reshape(-1, 640)
negative_train = torch.cat([word_train, format_train])
negative_all = torch.cat([current['word'].reshape(-1, 640), surrogate_word.reshape(-1, 640), current['format'][:, non_equals].reshape(-1, 640)])
XW, XN = word_train, format_train
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
minimum_training_bit_margin = float((ratios * true_bits).min())
assert minimum_training_bit_margin > 0
delta = min(.25, .25 * minimum_training_bit_margin)
designs = {}
for paired_top_bits in range(5):
    paired_bits = list(range(4 - paired_top_bits, 4))
    width = 14 + 3 * paired_top_bits
    label = f'mixed_top{paired_top_bits}_paired_{width}'
    bases, ups = [(1 + delta) * C, (1 - delta) * C], [C, C]
    transform = torch.zeros(4, width, dtype=torch.float64)
    transform[0, 0] = 1 / (2 * delta); transform[0, 1] = -1 / (2 * delta)
    transform[1:] = -transform[:1]
    bit_columns, quadratic_units = [], []
    column = 2
    for j in range(12):
        head, bit = divmod(j, 4)
        if bit in paired_bits:
            bases.extend([B[j] + delta * C, B[j] - delta * C]); ups.extend([C, C])
            transform[head + 1, column] = 2 ** bit / (4.5 * 2 * delta)
            transform[head + 1, column + 1] = -2 ** bit / (4.5 * 2 * delta)
            bit_columns.extend([j, j]); column += 2
        else:
            bases.append(B[j]); ups.append(C)
            bit_columns.append(j); quadratic_units.append((column, j)); column += 1
    assert column == width
    base_rows, up_rows = torch.stack(bases), torch.stack(ups)
    expected = bits[current_labels].reshape(-1, 12)[:, bit_columns]
    expected = torch.cat([torch.ones(len(expected), 2), expected], 1)
    designs[label] = {'base_gate_rows': base_rows, 'base_up_rows': up_rows,
        'transform': transform, 'expected_gate_signs': expected,
        'negative_training_base': negative_train @ base_rows.T,
        'paired_top_bits': paired_top_bits, 'bit_columns': bit_columns,
        'quadratic_units': quadratic_units}

negative_masses = [.01, .1, 1., 10., 100.]
positive_masses = [1., 10., 100., 1000., 10000.]
fixed_negative_metric_ridge = 1e-6
up_gain = 1e6
trials, selected = [], {}
for mass in negative_masses:
    covariance = (CW + mass * CN) / (1 + mass)
    rhs = -(mw + mass * mn) / (1 + mass)
    # Whiten the ordinary-negative covariance, then use one SVD of the
    # format-relative design. This avoids squaring its large carrier mode.
    covariance.diagonal().add_(fixed_negative_metric_ridge * covariance.diagonal().clamp_min(1e-15))
    factor = torch.linalg.cholesky((covariance + covariance.T) / 2)
    whitened_positive = torch.linalg.solve_triangular(factor, relative_positive.T, upper=False).T
    _, singular, right = torch.linalg.svd(whitened_positive / len(P) ** .5, full_matrices=False)
    whitened_rhs = torch.linalg.solve_triangular(factor, rhs[:, None], upper=False)[:, 0]
    projected = right @ whitened_rhs
    for positive_mass in positive_masses:
        whitened_off = right.T @ (projected / (1 + positive_mass * singular.square()))
        off = torch.linalg.solve_triangular(factor.T, whitened_off[:, None], upper=True)[:, 0]
        off_train = negative_train @ off
        off_max = float(off_train.max())
        for label, design in designs.items():
            rec = {'design': label, 'negative_format_mass': mass, 'relative_positive_mass': positive_mass,
                'OFF_training_maximum': off_max, 'programme_units': len(design['base_gate_rows']),
                'paired_top_bits': design['paired_top_bits'], 'eligible': False}
            if off_max >= -1e-8:
                rec['reason'] = 'OFF field does not separate every training negative'; trials.append(rec); continue
            negative_guard = .25 * float(positive_reference.min())
            bound = ((design['negative_training_base'] + negative_guard) / (-off_train[:, None])).max().clamp_min(0)
            off_scale = 1.001 * float(bound) + 1e-6
            unscaled = design['base_gate_rows'] + off_scale * off
            train_eq = P @ unscaled.T
            expected = bits[interval_y].reshape(-1, 12)[:, design['bit_columns']]
            expected = torch.cat([torch.ones(len(P), 2), expected], 1)
            positive = train_eq[expected > 0]
            negative = torch.cat([-train_eq[expected < 0], -(negative_train @ unscaled.T).flatten()])
            rec.update(OFF_scale=off_scale, maximum_training_relative_OFF=float((relative_positive @ off).abs().max()),
                       minimum_unscaled_training_positive=float(positive.min()),
                       minimum_unscaled_training_negative_magnitude=float(negative.min()))
            if min(float(positive.min()), float(negative.min())) <= 0:
                rec['reason'] = 'Hand-programme gate signs fail before scaling'; trials.append(rec); continue
            gate_gain = 1.001 * max(64 / float(positive.min()), 32 / float(negative.min()))
            # Hand-derived plateau on the actual positive Gate/C interval,
            # including the shared OFF field. Targets remain source bits only.
            trial_transform = design['transform'].clone()
            trial_up_rows = design['base_up_rows'].clone()
            quadratic_intervals = []
            positive_bits = bits[interval_y].reshape(-1, 12)
            for column, bit_column in design['quadratic_units']:
                head, bit = divmod(bit_column, 4)
                values = (train_eq[:, column] / positive_reference)[positive_bits[:, bit_column] > 0]
                lo, hi = float(values.min()), float(values.max())
                center, radius = (lo + hi) / 2, (hi - lo) / 2
                denominator = center * center - radius * radius / 2
                assert lo > 0 and denominator > 0
                trial_up_rows[column] = 2 * center * C - unscaled[column]
                trial_transform[head + 1, column] = 2 ** bit / (4.5 * denominator)
                quadratic_intervals.append({'source_place': head, 'bit': bit, 'unit_column': column,
                    'positive_gate_over_reference_minimum': lo, 'maximum': hi,
                    'center': center, 'radius': radius, 'output_denominator': denominator,
                    'single_bit_uniform_plateau_error_bound': radius * radius / (2 * denominator)})
            rec['conjugate_quadratic_intervals'] = quadratic_intervals
            G_real = gate_gain * unscaled
            unscaled_down = Wprogramme @ trial_transform
            down_scale = min(1., 1e6 / float(unscaled_down.abs().max()))
            transform = trial_transform * down_scale
            U_real = (up_gain / down_scale) * trial_up_rows
            # Positive homogeneous rescaling preserves every ideal GLU
            # product. This is an analytic weight-cap balance, not a trial.
            balance = min(1., (1e6 * (1 - 4 * torch.finfo(torch.float32).eps)) / float(G_real.abs().max()))
            G = (balance * G_real).float()
            U = (U_real / balance).float()
            down = (Wprogramme @ transform).float()
            rec.update(gate_gain=gate_gain * balance, up_gain=up_gain / down_scale / balance,
                       analytic_Gate_Up_balance=balance, unbalanced_gate_gain=gate_gain,
                       common_programme_down_scale=down_scale, maximum_gate_weight=float(G.abs().max()),
                       maximum_up_weight=float(U.abs().max()), maximum_programme_down_weight=float(down.abs().max()))
            if max(rec['maximum_gate_weight'], rec['maximum_up_weight'], rec['maximum_programme_down_weight']) > 1e6:
                rec['reason'] = 'A programme weight exceeds1e6'; trials.append(rec); continue
            word_gates = negative_all.float() @ G.T
            current_gates = current['format'][:, 8].float() @ G.T
            sign = design['expected_gate_signs']
            rec.update(maximum_actual_word_gate=float((current['word'].reshape(-1, 640).float() @ G.T).max()),
                maximum_surrogate_word_gate=float((surrogate_word.reshape(-1, 640).float() @ G.T).max()),
                maximum_all_negative_gate=float(word_gates.max()),
                minimum_current_positive_gate=float(current_gates[sign > 0].min()),
                maximum_current_negative_bit_gate=float(current_gates[sign < 0].max()),
                maximum_training_absolute_OFF=float((P @ off).abs().max()),
                maximum_all_current_relative_OFF=float(((current['format'][:, 8] @ off) / (current['format'][:, 8] @ C)).abs().max()))
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
                expected_bits = bits[y].reshape(-1, 12)[:, design['bit_columns']]
                expected_signs = torch.cat([torch.ones(len(X), 2), expected_bits], 1)
                signed = gate * expected_signs
                case_wrong = ~valid_reference
                case_wrong[valid_reference] = (got.round() != y[valid_reference]).any(1)
                case_bad = case_wrong | (signed <= 16).any(1)
                raw = features @ down.double().T
                reference_raw = effective @ Wprogramme.T
                diagnostics.append({'split': name, 'factual_current_mixtures': factual,
                    'finite_decoding': finite_decoding,
                    'invalid_reference_cases': int((~valid_reference).sum()),
                    'maximum_source_digit_error': float(error.max()) if finite_decoding else None,
                    'errors_per_place': error.amax(0).tolist() if finite_decoding else None,
                    'wrong_source_digits_with_valid_reference': int((got.round() != y[valid_reference]).sum()),
                    'source_cases_wrong_or_invalid': int(case_wrong.sum()),
                    'source_cases_wrong_invalid_or_unsaturated': int(case_bad.sum()),
                    'cases_with_unsaturated_or_wrong_gates': int((signed <= 16).any(1).sum()),
                    'cases': len(X),
                    'minimum_effective_reference': float(effective[:, 0].min()),
                    'minimum_absolute_signed_gate': float(signed.min()),
                    'wrong_gate_signs': int((signed <= 0).sum()),
                    'rounded_programme_Down_relative_error': float((raw - reference_raw).norm() / reference_raw.norm()) if reference_raw.norm() > 0 else None})
            rec['source_diagnostics'] = diagnostics
            worst = max(x['maximum_source_digit_error'] if x['finite_decoding'] else float('inf') for x in diagnostics)
            rec['strict_all_groups_eligible'] = bool(G.isfinite().all() and U.isfinite().all() and down.isfinite().all()
                and rec['maximum_all_negative_gate'] < -16
                and rec['minimum_current_positive_gate'] > 32 and rec['maximum_current_negative_bit_gate'] < -16
                and all(x['finite_decoding'] and x['minimum_effective_reference'] > 0 and x['minimum_absolute_signed_gate'] > 16
                    and x['wrong_source_digits_with_valid_reference'] == 0 and x['wrong_gate_signs'] == 0 for x in diagnostics)
                and worst < .15)
            current_only = diagnostics[:2]
            historical_cases = sum(x['cases'] for x in diagnostics[2:])
            historical_bad_cases = sum(x['source_cases_wrong_invalid_or_unsaturated'] for x in diagnostics[2:])
            # Explicit exploratory policy: current factual checks must all
            # pass; at most0.5% old ideal-routing cases may fail. This is
            # not an arithmetic pass or a claim about a fresh trajectory.
            rec['historical_cases'] = historical_cases
            rec['historical_bad_case_count'] = historical_bad_cases
            rec['current_only_exploratory_eligible'] = bool(G.isfinite().all() and U.isfinite().all() and down.isfinite().all()
                and rec['maximum_all_negative_gate'] < -16
                and rec['minimum_current_positive_gate'] > 32 and rec['maximum_current_negative_bit_gate'] < -16
                and all(x['finite_decoding'] and x['maximum_source_digit_error'] < .15
                    and x['minimum_effective_reference'] > 0 and x['minimum_absolute_signed_gate'] > 16
                    and x['source_cases_wrong_or_invalid'] == 0 and x['wrong_gate_signs'] == 0 for x in current_only)
                and historical_bad_cases <= .005 * historical_cases)
            rec['eligible'] = rec['strict_all_groups_eligible'] or rec['current_only_exploratory_eligible']
            rec['selection_policy'] = 'strict_all_groups' if rec['strict_all_groups_eligible'] else 'explicit_current_factual_exploratory_screen'
            trials.append(rec)
            current_worst = max(x['maximum_source_digit_error'] if x['finite_decoding'] else float('inf') for x in current_only)
            key = (not rec['strict_all_groups_eligible'], historical_bad_cases, current_worst, rec['maximum_gate_weight'])
            if rec['eligible'] and (label not in selected or key < selected[label][0]):
                selected[label] = key, G.clone(), U.clone(), down.clone(), transform.clone(), rec
    print(json.dumps({'negative_format_mass': mass, 'selected': {k: v[-1] for k, v in selected.items()}}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'source_field_key': args.field_key,
    'source_background_mass': entry['record']['background_mass'],
    'source_word_regularizer': entry['record']['word_regularizer'],
    'source_field_record': entry['record'],
    'hinge_width_rule': 'min(0.25,one quarter of the minimum training signed-bit/reference margin)',
    'minimum_training_source_bit_margin': minimum_training_bit_margin,
    'negative_format_mass_trials': negative_masses, 'relative_positive_mass_trials': positive_masses,
    'fixed_negative_metric_ridge_fraction': fixed_negative_metric_ridge,
    'hinge_half_width': delta, 'constant_pair_midpoint': 1., 'fixed_common_up_gain': up_gain,
    'OFF_scale_rule': '1.001times the minimum training-negative scale with a0.25minimum-positive-reference guard, plus1e-6',
    'gate_gain_rule': '1.001 times the exact training bound for positive64 and negative32 margins',
    'programme_down_scale_rule': 'Uniform analytic scale<=1 to keep programme Down coefficients<=1e6, with its reciprocal on Up to preserve the programme raw amplitude',
    'training_equals_rows': len(P),
    'OFF_fit': 'One generalized-covariance solve with positive outputs divided by their source reference; full SVD after whitening the original negative covariance; no exact training nullspace',
    'ordinary_negative_populations': ['actual_native_V_reader', 'recorded_A174_A176_native_feature_surrogate'],
    'equal_population_weight_in_word_covariance': True,
    'native_feature_calibration_parent': layout['base'],
    'training_word_windows_per_population': 192, 'validation_word_windows': 64, 'current_format_training_windows': 96, 'current_holdback_windows': 32,
    'historical_training_equals_per_group': 512,
    'quadratic_positive_bit_interval_low': low.tolist(), 'quadratic_positive_bit_interval_high': high.tolist(),
    'original_source_bit_interval_centers_diagnostic_only': mid.tolist(),
    'original_source_bit_interval_scales_diagnostic_only': quadratic_scale.tolist(),
    'quadratic_interval_target': 'Independent positive Gate/C intervals including OFF, never an operand-pair or sum',
    'paired_top_bit_count_designs': [0,1,2,3,4],
    'programme_unit_counts': [14,17,20,23,26],
    'quadratic_Up_rule': '2times training interval center times C, minus the complete unscaled Gate including OFF',
    'historical_caveat': 'Old states with ideal routing; component diagnostics only, not exact current trajectories',
    'analytic_Gate_Up_balance_rule': 'Scale Gate bymin(1,(1e6*(1-4float32epsilon))/maxabs(Gate)), and Up by its reciprocal; ideal product preserved, all actual margins rechecked',
    'exploratory_policy': 'Require all128current factual cases and32holdback cases correct, <0.15digit error and saturated; allow at most0.5percent bad or unsaturated cases across1536old ideal-routing diagnostics; no H1claim without full score',
    'scope': 'Programme rows only; native unit allocation, displaced-feature repair and full scoring remain necessary',
    'only_hand_programme_ReLU_ideal_no_native_nonlinear_replay': True,
    'no_model_checkpoint_gradients_iterative_refit_or_answers': True,
    'trials': trials, 'selected': {k: v[-1] for k, v in selected.items()},
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, source_base, fp, mp, wp, ep, ap, lp, cp] + check_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / ('artifacts/a227v1_conjugate_hybrid_binary_gates_' + args.field_key.replace('.', 'p').replace('-', 'm') + '.json'); atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'designs': {k: {'gate_rows': v[1], 'up_rows': v[2], 'programme_down_columns': v[3],
        'effective_feature_transform': v[4], 'record': v[5]} for k, v in selected.items()}, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
