"""Install a fully screened A355 native-Up repair on the compact A354 reader.

No fit or scale choice. Original programme rows and every other tensor stay
bit-identical to A354. Conditional component bounds still need a full score.
"""
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha

ap = ROOT / 'artifacts/a355v1_expanded_joint_native_up.pt'
fp = ap.with_suffix('.json')
ip = ROOT / 'artifacts/a356v1_recorded_linear_kernel_identity.json'
obj = torch.load(ap, weights_only=True)
report = json.loads(fp.read_text()); identity = json.loads(ip.read_text())
assert obj['report'] == report and report['selected'] is not None
selected = report['selected']
assert selected['eligible'] and selected['validation_restricted_matches'] >= report['baseline']['validation_restricted_matches'] + 16
assert selected['validation_certified_restricted_matches'] >= report['baseline']['validation_restricted_matches'] + 8
assert identity['all_available_controls_bit_exact_and_intervals_consistent']
assert not identity['native_Up_outputs_directly_observed']
full, fmt = report['full_vocabulary_validation'], report['format_validation']
assert full['candidate']['matches'] >= full['parent']['matches'] + 16
assert fmt['candidate']['matches'] == fmt['parent']['matches'] == 2560
assert fmt['candidate']['minimum_target_margin'] > 1
assert fmt['candidate']['minimum_certified_raw_margin'] > 0
assert fmt['maximum_raw_RMS_relative_change'] < .02
for evidence in (report, identity):
    for path, digest in {**evidence['sources'], **evidence['inputs']}.items():
        assert sha(ROOT / path) == digest, path

base = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
assert report['base_checkpoint'] == str(base.relative_to(ROOT))
cp = base.parent / 'circuit.pt'
bm = json.loads(base.with_suffix('.manifest.json').read_text())
assert sha(base) == bm['delta_sha256'] == '20daec4dbeb0d87870270c7cbb1e6c0ea2f7b624369d3b8c54b580a25489f399'
for path, digest in bm['sources'].items(): assert sha(ROOT / path) == digest, path
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(cp, weights_only=True))
un = 'model.layers.17.mlp.up_proj.weight'
units = torch.tensor(selected['units']); programme = c['units'].long()
assert len(units) == selected['native_unit_count'] and not torch.isin(units, programme).any()
new = obj['up']; old = ch[un]
assert new.dtype == torch.float32 and new.shape == old.shape and new.isfinite().all() and new.abs().max() <= 1e6
remaining = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert torch.equal(new[remaining], old[remaining]) and torch.equal(new[programme], old[programme])
ch[un] = new

audit_path = ROOT / 'artifacts/a354v1_edit_efficient_plus_reader_edit_category_audit.json'
parent_audit = json.loads(audit_path.read_text()); assert parent_audit['delta_sha256'] == bm['delta_sha256']
parent_rows = {row['tensor']: row for row in parent_audit['per_tensor']}
categories = {}
for name, count in bm['per_tensor'].items():
    if name == un: continue
    row = parent_rows[name]; assert row['changed_scalars'] == count
    categories[name] = {'arithmetic_and_circuit_accommodation': row['arithmetic_and_circuit_accommodation'],
                        'native_text_repair': row['native_only_text_repair']}
mask = new.view(torch.int32) != h[un].view(torch.int32)
arithmetic_count = int(mask[programme].sum())
categories[un] = {'arithmetic_programme_Up': arithmetic_count, 'native_text_repair': int(mask.sum()) - arithmetic_count}
c['expanded_native_up_a357'] = {'analysis': report, 'linear_identity_audit': identity,
    'authoritative_native_Up_units_layer17': units, 'all_arithmetic_Up_rows_unchanged': True,
    'legacy_native_Up_note': 'The listed native17 Up rows now follow A355. Their unchanged Gates and all other programme/native weights follow A354.',
    'conditional_factor_scope': identity['conditional_inference_assumption']}
out = ROOT / 'experiments/a357v1_expanded_native_up/delta.pt'
assert not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
sources = set(bm['sources']); inputs = set()
for evidence in (report, identity):
    sources.update(evidence['sources']); inputs.update(evidence['inputs'])
manifest = save(out, h, ch, sources=[Path(__file__)] + [ROOT / p for p in sources],
    inputs=[base, cp, base.with_suffix('.manifest.json'), ap, fp, ip, audit_path] + [ROOT / p for p in inputs],
    arguments={}, description=__doc__, categories=categories,
    extra={'selected': selected, 'full_vocabulary_validation': full, 'format_validation': fmt,
        'changed_tensors': [un], 'all_other_A354_weights_bit_identical': True,
        'all_programme_Up_rows_bit_identical': True,
        'conditional_linear_identity_assumption': identity['conditional_inference_assumption'],
        'component_bounds_are_not_full_model_execution_certificates': True})
assert manifest['changed_scalars'] == selected['total_changed_scalars']
assert sum(sum(v.values()) for v in categories.values()) == manifest['changed_scalars']
per_tensor = []
for name, count in manifest['per_tensor'].items():
    native_count = categories[name].get('native_text_repair', 0)
    per_tensor.append({'tensor': name, 'changed_scalars': count,
        'arithmetic_and_circuit_accommodation': count - native_count, 'native_only_text_repair': native_count})
audit = {'checkpoint': str(out.relative_to(ROOT)), 'delta_sha256': manifest['delta_sha256'],
    'changed_scalars': manifest['changed_scalars'], 'max_abs_edited_value': manifest['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation': sum(r['arithmetic_and_circuit_accommodation'] for r in per_tensor),
    'native_only_text_repair': sum(r['native_only_text_repair'] for r in per_tensor), 'per_tensor': per_tensor,
    'scope': 'Inherited audited masks except native17 Up; disjoint actual Up masks. Purpose bookkeeping, not causal attribution.',
    'source_sha256': sha(__file__), 'inputs': {str(out.relative_to(ROOT)): sha(out), str(audit_path.relative_to(ROOT)): sha(audit_path)}}
atomic_json(ROOT / 'artifacts/a357v1_expanded_native_up_edit_category_audit.json', audit)
print(json.dumps({'delta_sha256': manifest['delta_sha256'], 'changed_scalars': manifest['changed_scalars'],
    'arithmetic': audit['arithmetic_and_circuit_accommodation'], 'native': audit['native_only_text_repair']}), flush=True)
