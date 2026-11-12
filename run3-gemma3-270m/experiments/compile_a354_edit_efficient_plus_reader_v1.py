"""Install the fixed realized-key plus reader on A259 for one full test.

A352 supplies Q/K and query gains; A326 supplies four literal-digit Up rows.
A353 bounds the complete local reader. The original strict decoder rejection
and the changed-word-trajectory uncertainty remain disclosed.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, atomic_json, sha
qp = ROOT / 'artifacts/a352v1_edit_efficient_A_reader.pt'
dp = ROOT / 'artifacts/a326v1_fixed_plus_decoder_diagnostic.pt'
fp = ROOT / 'artifacts/a353v1_edit_efficient_plus_reader_composition.json'
qobj, dobj = [torch.load(p, weights_only=True) for p in (qp, dp)]
f = json.loads(fp.read_text())
assert qobj['report']['original_A343_diagnostic_bit_exact'] and qobj['report']['original_ten_percent_screen_rejection_retained']
assert qobj['report']['selected']['eligible'] and qobj['report']['selected']['target_source_margin'] == 12.
assert dobj['report']['original_A324_diagnostic_bit_exact'] and dobj['report']['selected'] is None
assert not dobj['report']['fixed_diagnostic']['eligible']
assert f['eligible_for_declared_exploratory_full_score']
assert all(rec['passes'] and rec['maximum_register_digit_error_bound'] < .5 for rec in f['format_reports'])
reports = [qobj['report'], dobj['report'], f]
# Include the original intermediate reports for complete transitive provenance.
for path in ('artifacts/a323v1_causal_metric_normalized_key.json',
             'artifacts/a324v1_independent_plus_decoder.json',
             'artifacts/a327v1_plus_reader_composition.json'):
    reports.append(json.loads((ROOT / path).read_text()))
for report in reports:
    for path, digest in {**report['sources'], **report['inputs']}.items():
        assert sha(ROOT / path) == digest, path
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
cp = base.parent / 'circuit.pt'; bm = json.loads(base.with_suffix('.manifest.json').read_text())
assert bm['delta_sha256'] == sha(base) == '39f6c03da1c8b7d38b2223e59e5f680b92627808711db109a0b456c3cd81f350'
for path, digest in bm['sources'].items(): assert sha(ROOT / path) == digest, path
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(cp, weights_only=True))
stage = c['native_a_stage']; units = stage['units'].long()
assert units.tolist() == [2021, 320, 1564, 141]
qn, gn, kn, un = [f'model.layers.3.{name}' for name in ('self_attn.q_proj.weight', 'self_attn.q_norm.weight', 'self_attn.k_proj.weight', 'mlp.up_proj.weight')]
assert set(qobj['tensors']) == {qn, gn, kn}
assert f['inputs'][str(qp.relative_to(ROOT))] == sha(qp)
assert f['inputs'][str(dp.relative_to(ROOT))] == sha(dp)
values = dict(qobj['tensors'])
up = ch[un].clone(); up[units] = dobj['up']; values[un] = up
for name, value in values.items():
    assert value.dtype == torch.float32 and value.shape == h[name].shape
    assert value.isfinite().all() and value.abs().max() <= 1e6
    ch[name] = value
# All other tensors are still the exact parent objects returned by expand.
audit_path = ROOT / 'artifacts/a259v1_low_rank_copy_query_on_a240_edit_category_audit.json'
audit = json.loads(audit_path.read_text()); assert audit['delta_sha256'] == bm['delta_sha256']
rows = {row['tensor']: row for row in audit['per_tensor']}
categories = {}
for name, count in bm['per_tensor'].items():
    if name in values: continue
    row = rows[name]; assert row['changed_scalars'] == count
    categories[name] = {'arithmetic_and_circuit_accommodation': row['arithmetic_and_circuit_accommodation'],
                        'native_text_repair': row['native_only_text_repair']}
channels = torch.tensor([4, 132]); ordinary = torch.arange(256)[~torch.isin(torch.arange(256), channels)]
mask = (values[qn].view(torch.int32) != h[qn].view(torch.int32)).reshape(4, 256, 640)
categories[qn] = {'arithmetic_query_with_native_accommodation': int(mask[:, channels].sum()),
                  'native_query_accommodation': int(mask[:, ordinary].sum())}
for name in (gn, kn):
    categories[name] = {'arithmetic_routing': int((values[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
categories[un] = {'arithmetic_features': int((up.view(torch.int32) != h[un].view(torch.int32)).sum())}
stage['reader']['frame'] = dobj['frame']
stage['reader']['piece']['source_positions'] = [3, 2, 1, 4]
stage['reader']['piece']['lags'] = [5, 6, 7, 4]
stage['reader']['piece']['reference'] = 'plus sign at offset-4'
stage['piece']['common_up_gain'] = dobj['report']['fixed_diagnostic']['common_Up_gain']
stage['piece']['anchor_up_gain'] = dobj['report']['fixed_diagnostic']['common_Up_gain']
stage['reader']['native_key_query_layer3']['selected_channels'] = channels
c['edit_efficient_plus_reader_a354'] = {'query': qobj['report'], 'decoder': dobj['report'], 'composition': f,
    'authoritative_query_programme_channels': channels, 'source_positions': [3, 2, 1, 4],
    'original_strict_decoder_rejection_retained': True, 'original_ten_percent_query_screen_rejection_retained':True, 'edit_efficiency_review_only':True,
    'legacy_query_metadata_note': 'A354 is authoritative for layer3 Q/K/qnorm and four decoder Up rows. Earlier A237/A240 native-query repairs and 64-channel programme metadata are inactive.',
    'changed_word_trajectory_unobserved_until_official_score': True}
extra = {'query_selection': qobj['report']['selected'], 'fixed_decoder_diagnostic': dobj['report']['fixed_diagnostic'],
    'composition_review': f, 'changed_tensor_names': list(values), 'all_other_A259_weights_bit_identical': True,
    'exploratory_full_test': True, 'original_ten_percent_query_screen_rejection_retained':True, 'freed_weight_edits':389117, 'original_strict_decoder_screen_failed': True,
    'universal_word_Gate_bound_inconclusive': True, 'purpose_categories_are_bookkeeping_not_causal_ablation': True}
out = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
assert not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
sources = set(bm['sources']); sources.add('experiments/compile_a330_strengthened_plus_reader_v1.py'); inputs = set()
for report in reports:
    sources.update(report['sources']); inputs.update(report['inputs'])
manifest = save(out, h, ch, sources=[Path(__file__)] + [ROOT / p for p in sources],
    inputs=[base, cp, base.with_suffix('.manifest.json'), qp, qp.with_suffix('.json'), dp, dp.with_suffix('.json'), fp, audit_path]
           + [ROOT / p for p in inputs], arguments={}, description=__doc__, extra=extra, categories=categories)
assert manifest['changed_scalars'] == f['changed_scalars'] == 4589583
assert sum(sum(v.values()) for v in categories.values()) == manifest['changed_scalars']
# Emit a disjoint edit audit directly from the actual compiler masks.
per_tensor = []
for name, count in manifest['per_tensor'].items():
    groups = categories[name]
    native = sum(value for key, value in groups.items() if key in ('native_text_repair', 'native_query_accommodation'))
    per_tensor.append({'tensor': name, 'changed_scalars': count,
                       'arithmetic_and_circuit_accommodation': count-native, 'native_only_text_repair': native})
a = {'checkpoint': str(out.relative_to(ROOT)), 'delta_sha256': manifest['delta_sha256'],
    'changed_scalars': manifest['changed_scalars'], 'max_abs_edited_value': manifest['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation': sum(r['arithmetic_and_circuit_accommodation'] for r in per_tensor),
    'native_only_text_repair': sum(r['native_only_text_repair'] for r in per_tensor), 'per_tensor': per_tensor,
    'scope': 'Parent index-audited labels for untouched tensors; actual compiler masks for changed tensors; bookkeeping, not causal attribution',
    'source_sha256': sha(__file__), 'inputs': {str(out.relative_to(ROOT)): sha(out), str(audit_path.relative_to(ROOT)): sha(audit_path)}}
atomic_json(ROOT / 'artifacts/a354v1_edit_efficient_plus_reader_edit_category_audit.json', a)
print(json.dumps({'sha256': manifest['delta_sha256'], 'changed_scalars': manifest['changed_scalars'],
    'arithmetic': a['arithmetic_and_circuit_accommodation'], 'native': a['native_only_text_repair']}), flush=True)
