"""Reconcile the compressed layer2 manifest with inherited purpose labels.

This reads only existing metadata and hashes. It does not execute a model or
claim a causal attribution for overlapping query repairs.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

base = ROOT / 'experiments/a313v1_duplicate_reference_compression/delta.pt'
mp = base.with_suffix('.manifest.json')
pp = ROOT / 'artifacts/a259v1_low_rank_copy_query_on_a240_edit_category_audit.json'
parent = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
m, prior = [json.loads(p.read_text()) for p in (mp, pp)]
assert m['delta_sha256'] == sha(base)
assert prior['delta_sha256'] == sha(parent)
assert m['extra']['all_other_A259_weights_bit_identical']
names = [f'model.layers.2.mlp.{part}_proj.weight' for part in ('gate', 'up', 'down')]
assert set(names) == set(m['extra']['changed_tensor_names'])
rows = []
for previous in prior['per_tensor']:
    row = dict(previous); name = row['tensor']
    if name in names:
        category = m['categories'][name]
        row['native_only_text_repair'] = category.get('native_text_repair', 0)
        row['arithmetic_and_circuit_accommodation'] = sum(category.values()) - row['native_only_text_repair']
        row['changed_scalars'] = sum(category.values())
        row['overlap_note'] = 'Four arithmetic feature units; unit727 restored natively; labels from the A313 compiler masks.'
    assert row['changed_scalars'] == m['per_tensor'][name]
    rows.append(row)
report = {'checkpoint': str(base.relative_to(ROOT)), 'delta_sha256': sha(base),
    'changed_scalars': m['changed_scalars'], 'max_abs_edited_value': m['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation': sum(r['arithmetic_and_circuit_accommodation'] for r in rows),
    'native_only_text_repair': sum(r['native_only_text_repair'] for r in rows),
    'category_interpretation': 'Inherited index-audited A259 labels outside layer2 Gate/Up/Down; replacement labels use the immutable A313 compiler masks. Q3/Q7 native repairs overlap arithmetic labels. These are disjoint bookkeeping labels, not causal allocations.',
    'scope': 'Metadata reconciliation against the compiler-verified unchanged parent tensors; no new sparse-index comparison is claimed.',
    'per_tensor': rows, 'immutable_manifest_unchanged': True,
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, parent, mp, pp)}, 'source_sha256': sha(__file__)}
assert sum(r['changed_scalars'] for r in rows) == report['changed_scalars']
assert report['arithmetic_and_circuit_accommodation'] + report['native_only_text_repair'] == report['changed_scalars']
out = ROOT / 'artifacts/a313v1_duplicate_reference_compression_edit_category_audit.json'
temporary = out.with_suffix('.json.tmp')
temporary.write_text(json.dumps(report, indent=2) + '\n'); temporary.replace(out)
print(json.dumps({k: report[k] for k in ('changed_scalars', 'arithmetic_and_circuit_accommodation', 'native_only_text_repair')}))
