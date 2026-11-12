"""Quantize selected independent source bits within the old analog interface.

The old common format Gate supplies a shared reference. Symmetric shifts of
existing screened bit Gate centers by that reference make each pair's
difference exact in the ReLU saturation ideal. Other bits remain linear.
No new fit or operand-answer target is used.
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


gate_path = ROOT / ('artifacts/a224v1_union_text_binary_gates_' + args.field_key.replace('.', 'p').replace('-', 'm') + '.pt')
gate_object = torch.load(gate_path, weights_only=True)
assert gate_object['report']['source_field_key'] == args.field_key
assert gate_object['report']['inputs'][str(fp.relative_to(ROOT))] == sha(fp)
original = gate_object['designs']['paired_hinge_26']
assert original['record']['strict_all_groups_eligible']
parent_gate_rows = original['gate_rows'].double()
centers = (parent_gate_rows[2::2] + parent_gate_rows[3::2]) / 2
assert centers.shape == (12, 640)
# Quiet fields classify bits; the already-measured best analog field carries
# the unquantized bits. Both target the same independent source tokens.
analog_key = 'w1e-08_b1'
analog_entry = catalog['alternatives'][analog_key]
analog_rows = analog_entry['input_rows'].double()
C, B = analog_rows[12], analog_rows[:12]
common_gate = old_gate_rows[0].double()
P = torch.cat([current['format'][:96, 8], historical[0], historical[1]])
training_labels = torch.cat([current_labels[:96], historical_labels[0], historical_labels[1]])
training_bits = bits[training_labels].reshape(-1, 12)
reference_gate = P @ common_gate
assert reference_gate.min() > 32
non_equals = torch.tensor([j for j in range(14) if j != 8])
negative_train = torch.cat([current['word'][:192].reshape(-1, 640), surrogate_word[:192].reshape(-1, 640), current['format'][:96, non_equals].reshape(-1, 640)])
negative_all = torch.cat([current['word'].reshape(-1, 640), surrogate_word.reshape(-1, 640), current['format'][:, non_equals].reshape(-1, 640)])
negative_common = negative_train @ common_gate
up_gain = float(stage['piece']['common_up_gain']); assert up_gain == 1e6
structures = {
    'units_bit0': [0],
    'units_bit1': [1],
    'units_bit2': [2],
    'units_top1': [3],
    'units_top2': [2,3],
    'units_top3': [1,2,3],
    'units_all4': [0,1,2,3],
    'units_and_tens_top1': [3,7],
}
trials, selected = [], {}
for label, quantized in structures.items():
    selected_centers = centers[quantized]
    signed_relative = (P @ selected_centers.T) * training_bits[:, quantized] / reference_gate[:, None]
    negative_centers = negative_train @ selected_centers.T
    if signed_relative.min() <= 0 or negative_centers.max() >= 0:
        trials.append({'design': label, 'quantized_source_bit_columns': quantized, 'eligible': False,
            'reason': 'Existing bit centers do not have all required training signs'})
        continue
    source_shift_bound = float(signed_relative.min())
    negative_shift_bound = float((-negative_centers / negative_common.abs()[:, None].clamp_min(1e-30)).min())
    shift = min(.25, .25 * source_shift_bound, .25 * negative_shift_bound)
    assert shift > 0
    gates, ups = [common_gate], [C]
    transform_columns = [torch.tensor([1.,0,0,0], dtype=torch.float64)]
    unit_bit = [-1]
    for head in range(3):
        unquantized = [bit for bit in range(4) if 4 * head + bit not in quantized]
        if unquantized:
            analog = sum((2 ** bit / 9) * B[4 * head + bit] for bit in unquantized)
            analog += (sum(2 ** bit for bit in unquantized) / 9 - 1) * C
            gates.append(common_gate); ups.append(analog)
            col = torch.zeros(4, dtype=torch.float64); col[head + 1] = 1
            transform_columns.append(col); unit_bit.append(-1)
        else:
            transform_columns[0][head + 1] = -1
    for j in quantized:
        head, bit = divmod(j, 4)
        for sign in (1., -1.):
            gates.append(centers[j] + sign * shift * common_gate); ups.append(C)
            col = torch.zeros(4, dtype=torch.float64)
            col[head + 1] = sign * 2 ** bit / (4.5 * 2 * shift)
            transform_columns.append(col); unit_bit.append(j)
    G = torch.stack(gates)
    transform = torch.stack(transform_columns, 1)
    unscaled_down = Wprogramme @ transform
    down_scale = min(1., 1e6 / float(unscaled_down.abs().max()))
    transform *= down_scale
    U = torch.stack(ups) * (up_gain / down_scale)
    balance = min(1., (1e6 * (1 - 4 * torch.finfo(torch.float32).eps)) / float(G.abs().max()))
    G = (G * balance).float(); U = (U / balance).float()
    down = (Wprogramme @ transform).float()
    rec = {'design': label, 'programme_units': len(G), 'quantized_source_bit_columns': quantized,
        'shared_common_gate_shift': shift, 'training_source_shift_bound': source_shift_bound,
        'training_negative_shift_bound': negative_shift_bound, 'fixed_common_up_gain': up_gain,
        'uniform_programme_down_scale': down_scale, 'analytic_Gate_Up_balance': balance,
        'maximum_gate_weight': float(G.abs().max()), 'maximum_up_weight': float(U.abs().max()),
        'maximum_programme_down_weight': float(down.abs().max()), 'eligible': False}
    if max(rec['maximum_gate_weight'], rec['maximum_up_weight'], rec['maximum_programme_down_weight']) > 1e6:
        rec['reason'] = 'Weight cap'; trials.append(rec); continue
    rec['maximum_all_negative_gate'] = float((negative_all.float() @ G.T).max())
    diagnostics = []
    groups = [('current128', current['format'][:, 8], current_labels),
              ('current_holdback32', current['format'][96:, 8], current_labels[96:])]
    groups += [(name, X, y) for name, X, y in zip(('historical_a', 'historical_stress', 'historical_c'), historical, historical_labels)]
    for name, X, y in groups:
        g = (X.float() @ G.T).double(); u = (X.float() @ U.T).double()
        features = g.clamp_min(0) * u
        code = features @ transform.T
        valid = code[:, 0] > 0
        finite = bool(valid.all() and code.isfinite().all())
        got = 4.5 * code[valid, 1:] / code[valid, :1] + 4.5
        errors = (got - y[valid]).abs()
        expected = torch.ones_like(g)
        source_bits = bits[y].reshape(-1, 12)
        for col, bit in enumerate(unit_bit):
            if bit >= 0: expected[:, col] = source_bits[:, bit]
        signed_gate = g * expected
        diagnostics.append({'split': name, 'finite_decoding': finite,
            'maximum_source_digit_error': float(errors.max()) if finite else None,
            'maximum_errors_by_place': errors.amax(0).tolist() if finite else None,
            'wrong_source_digits': int((got.round() != y[valid]).sum()),
            'invalid_reference_cases': int((~valid).sum()),
            'minimum_signed_gate': float(signed_gate.min()),
            'unsaturated_or_wrong_gate_cases': int((signed_gate <= 16).any(1).sum()),
            'maximum_programme_Down_identity_error': float((features @ down.double().T - code @ Wprogramme.T).abs().max())})
    current_error = max(d['maximum_source_digit_error'] if d['finite_decoding'] else float('inf') for d in diagnostics[:2])
    historical_error = max(d['maximum_source_digit_error'] if d['finite_decoding'] else float('inf') for d in diagnostics[2:])
    rec['source_diagnostics'] = diagnostics
    rec['maximum_current_digit_error'] = current_error if current_error != float('inf') else None
    rec['maximum_historical_digit_error'] = historical_error if historical_error != float('inf') else None
    rec['eligible'] = bool(G.isfinite().all() and U.isfinite().all() and down.isfinite().all()
        and rec['maximum_all_negative_gate'] < -16 and current_error < .25 and historical_error < .4
        and all(d['finite_decoding'] and d['wrong_source_digits'] == 0 and d['minimum_signed_gate'] > 16 for d in diagnostics))
    trials.append(rec)
    if rec['eligible']:
        selected[label] = {'gate_rows': G, 'up_rows': U, 'programme_down_columns': down,
            'effective_feature_transform': transform, 'record': rec}
    print(json.dumps(rec), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'source_field_key': args.field_key,
    'source_catalog': str(fp.relative_to(ROOT)), 'parent_bit_gate_artifact': str(gate_path.relative_to(ROOT)),
    'structures': structures, 'programme_unit_counts': [6,6,6,6,8,10,11,8],
    'fixed_analog_source_field_key': analog_key,
    'analog_field_choice': 'Previously measured smallest current scalar source error in A228; no new fit',
    'new_fit_or_scale_trials': [],
    'common_shift_rule': 'min(0.25,one quarter of the minimum training signed center/common-Gate ratio,one quarter of the minimum negative-center/absolute-common-Gate ratio)',
    'hand_identity': 'relu(center+shift*common)-relu(center-shift*common)=2*shift*common for each positive source bit and0for each negative bit; multiply by shared C Up',
    'analog_identity': 'Unquantized signed bits/9 plus their exact constant offset; quantized bits contribute2^bit/4.5 only when positive',
    'source_guard': 'Current errors<0.25, historical<0.4, no wrong/invalid digits, every source gate>16 in its correct sign, both word populations and non-equals<-16',
    'training_word_windows_per_population': 192, 'validation_word_windows_per_population': 64,
    'current_format_training_windows': 96, 'current_format_holdback_windows': 32,
    'historical_training_groups': ['historical_a','historical_stress'], 'historical_holdback_group': 'historical_c',
    'no_native_activation_replay_gradients_iterative_fitting_or_answers': True,
    'scope': 'Hand programme rows only; native allocation and official full score still required',
    'trials': trials, 'selected': {label: value['record'] for label, value in selected.items()},
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, base.parent / 'circuit.pt', source_base, fp, gate_path, mp, wp, ep, ap, lp, cp] + check_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / ('artifacts/a231v1_selective_bit_quantization_' + args.field_key.replace('.', 'p').replace('-', 'm') + '.json')
atomic_json(out, report)
if selected: atomic_torch(out.with_suffix('.pt'), {'designs': selected, 'report': report})
print(json.dumps({'selected_designs': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
