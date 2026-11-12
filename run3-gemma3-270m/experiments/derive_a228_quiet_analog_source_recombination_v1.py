"""Recombine existing independent source-bit fields into four analog rows.

This adds no fit or trial constant: binary expansion gives the normalized
source digit linearly. Existing common Gate and all native tensors stay fixed.
Only source tokens are decoded in analysis; no addition answer is computed.
"""
import argparse, json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'; ch = expand(h, base)
fp = ROOT / 'artifacts/a221v1_native_binary_field_catalog.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
catalog, mix, w, e = [torch.load(p, weights_only=True, mmap=True) for p in (fp, mp, wp, ep)]
source_base = ROOT / catalog['report']['base_checkpoint']; reference = expand(h, source_base)
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


common_up_gain = float(stage['piece']['common_up_gain']); assert common_up_gain == 1e6
G = old_gate_rows.double()
word_gate_max = float((current['word'].reshape(-1, 640).float() @ G.float().T).max())
surrogate_gate_max = float((surrogate_word.reshape(-1, 640).float() @ G.float().T).max())
current_gate_min = float((current['format'][:, 8].float() @ G.float().T).min())
trials, best, alternatives = [], None, {}
for key, entry in catalog['alternatives'].items():
    source_rows = entry['input_rows'].double()
    rows = torch.empty(4, 640, dtype=torch.float64); rows[0] = source_rows[12]
    for head in range(3):
        rows[head + 1] = ((2 ** torch.arange(4, dtype=torch.float64))[:, None] * source_rows[4 * head:4 * head + 4]).sum(0) / 9 + (2 / 3) * source_rows[12]
    Up = (common_up_gain * rows).float()
    actual_frame = (Up.double() / common_up_gain).T * gp[:, None]
    diagnostics = []
    groups = [('current128', current['format'][:, 8], current_labels),
              ('current_holdback32', current['format'][96:, 8], current_labels[96:])]
    groups += [(name, X, y) for name, X, y in zip(('historical_a', 'historical_stress', 'historical_c'), historical, historical_labels)]
    for name, X, y in groups:
        code = (X.float() @ Up.T).double()
        positive_reference = code[:, 0] > 0
        finite = bool(positive_reference.all() and code.isfinite().all())
        got = 4.5 * code[:, 1:] / code[:, :1] + 4.5
        errors = (got - y).abs()
        diagnostics.append({'split': name, 'finite_positive_reference': finite,
            'minimum_reference': float(code[:, 0].min() / common_up_gain),
            'maximum_source_digit_error': float(errors.max()) if finite else None,
            'maximum_error_by_place': errors.amax(0).tolist() if finite else None,
            'wrong_source_digits': int((got.round() != y).sum()), 'digits': y.numel()})
    current_error = max(d['maximum_source_digit_error'] if d['finite_positive_reference'] else float('inf') for d in diagnostics[:2])
    historical_error = max(d['maximum_source_digit_error'] if d['finite_positive_reference'] else float('inf') for d in diagnostics[2:])
    rec = {'source_field_key': key, 'word_regularizer': entry['record']['word_regularizer'],
        'background_mass': entry['record']['background_mass'], 'common_up_gain': common_up_gain,
        'maximum_up_weight': float(Up.abs().max()), 'source_diagnostics': diagnostics,
        'maximum_current_digit_error': current_error if current_error != float('inf') else None,
        'maximum_historical_digit_error': historical_error if historical_error != float('inf') else None,
        'maximum_actual_word_gate': word_gate_max, 'maximum_surrogate_word_gate': surrogate_gate_max,
        'minimum_current_equals_gate': current_gate_min}
    rec['eligible'] = bool(Up.isfinite().all() and Up.abs().max() <= 1e6 and current_error < .25
        and historical_error < .4 and all(d['wrong_source_digits'] == 0 and d['finite_positive_reference'] for d in diagnostics)
        and word_gate_max < -16 and surrogate_gate_max < -16 and current_gate_min > 32)
    trials.append(rec)
    if rec['eligible']:
        alternatives[key] = {'up_rows': Up.clone(), 'reader_frame': actual_frame[:, [1,2,3,0]].clone(), 'record': rec}
        sortkey = (current_error, historical_error)
        if best is None or sortkey < best[0]: best = sortkey, key, rec
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'source_catalog': str(fp.relative_to(ROOT)), 'source_field_keys': list(catalog['alternatives']),
    'construction': 'Dnormalized=sum(2^bit*signed_bit)/9+(2/3)*C, with C unchanged; same original independent source-token fit by linearity',
    'new_scale_trials': [], 'fixed_common_up_gain': common_up_gain,
    'source_guards': 'Current source errors<0.25, historical errors<0.4, no wrong/invalid digits, existing common Gate off on both word populations',
    'all_native_and_other_programme_weights_unchanged': True,
    'no_new_fit_model_execution_native_activation_replay_gradients_or_answers': True,
    'trials': trials, 'selected_key': None if best is None else best[1],
    'selected': None if best is None else best[2], 'eligible_field_keys': list(alternatives),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, base.parent / 'circuit.pt', source_base, fp, mp, wp, ep, ap, lp, cp] + check_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a228v1_quiet_analog_source_recombination.json'; atomic_json(out, report)
if alternatives: atomic_torch(out.with_suffix('.pt'), {'alternatives': alternatives, 'report': report})
print(json.dumps({'selected': report['selected'], 'eligible_fields': report['eligible_field_keys'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
