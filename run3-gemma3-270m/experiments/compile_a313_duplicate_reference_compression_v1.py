"""Install an exact four-unit B reader with one restored native MLP unit.

Two original reference units have identical Gate/Up rows and disjoint Down
supports. Their programme maps merge exactly; native behavior is rebuilt
once from original HOST targets. Full scoring tests the changed trajectory.
"""
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a313v1_duplicate_reference_compression.pt'
format_path = ROOT / 'artifacts/a314v1_compressed_reference_format.json'
obj = torch.load(source, weights_only=True)
analysis = obj['report']; selected = analysis['selected']
format_report = json.loads(format_path.read_text())
assert selected and selected['eligible_local'] and format_report['eligible']
assert format_report['selected_compression'] == selected
assert format_report['inputs'][str(source.relative_to(ROOT))] == sha(source)
assert analysis == json.loads(source.with_suffix('.json').read_text())
for report in (analysis, format_report):
    for path, digest in {**report['sources'], **report['inputs']}.items():
        assert sha(ROOT / path) == digest, path
base = ROOT / analysis['base_checkpoint']
bm = json.loads(base.with_suffix('.manifest.json').read_text())
assert bm['delta_sha256'] == sha(base) == '39f6c03da1c8b7d38b2223e59e5f680b92627808711db109a0b456c3cd81f350'
for path, digest in bm['sources'].items():
    assert sha(ROOT / path) == digest, path
h = host_state(); ch = expand(h, base)
names = [f'model.layers.2.mlp.{part}_proj.weight' for part in ('gate', 'up', 'down')]
assert set(obj['weights']) == set(names)
units = torch.tensor(selected['programme_units'])
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert units.tolist() == [80, 342, 529, 1634]
assert selected['restored_native_unit'] == 727 and selected['retained_reference_unit'] == 529
assert torch.equal(ch[names[0]][529], ch[names[0]][727])
assert torch.equal(ch[names[1]][529], ch[names[1]][727])
expected_programme = ch[names[2]][:, units].clone()
expected_programme[:, (units == 529).nonzero().item()] += ch[names[2]][:, 727]
assert torch.equal(obj['weights'][names[2]][:, units], expected_programme)
for name in names[:2]:
    assert torch.equal(obj['weights'][name][units], ch[name][units])
    assert torch.equal(obj['weights'][name][native], h[name][native])

categories = copy.deepcopy(bm['categories'])
audit_path = ROOT / 'artifacts/a259v1_low_rank_copy_query_on_a240_edit_category_audit.json'
audit = json.loads(audit_path.read_text())
assert audit['delta_sha256'] == bm['delta_sha256']
for row in audit['per_tensor']:
    if row['tensor'] not in categories:
        assert bm['per_tensor'][row['tensor']] == row['changed_scalars']
        categories[row['tensor']] = {
            'arithmetic_and_circuit_accommodation': row['arithmetic_and_circuit_accommodation'],
            'native_text_repair': row['native_only_text_repair']}
for name in names:
    value = obj['weights'][name]
    assert value.dtype == torch.float32 and value.shape == h[name].shape
    ch[name] = value
    changed = value.view(torch.int32) != h[name].view(torch.int32)
    if name.endswith('down_proj.weight'):
        categories[name] = {'arithmetic_programme': int(changed[:, units].sum()),
                            'native_text_repair': int(changed[:, native].sum())}
    else:
        categories[name] = {'arithmetic_features': int(changed.sum())}

parent_circuit = base.parent / 'circuit.pt'
c = copy.deepcopy(torch.load(parent_circuit, weights_only=True))
c['duplicate_reference_compression_a313'] = {
    'layer': 2, 'original_programme_units': [80, 342, 529, 727, 1634],
    'current_programme_units': units, 'restored_native_unit': 727,
    'retained_reference_unit': 529, 'analysis': analysis, 'format_bound': format_report}
c['legacy_layer2_unit_indices_note'] = (
    'duplicate_reference_compression_a313 is authoritative for layer2. '
    'The retained unit529 writes both the original529 and727 programme Down columns; '
    'unit727 now has HOST Gate/Up and reconstructed native Down. Legacy five-unit metadata is inactive.')
extra = {'local_selection': selected, 'format_bound': format_report,
         'changed_tensor_names': names, 'all_other_A259_weights_bit_identical': True,
         'purpose_categories_are_bookkeeping_not_causal_ablation': True,
         'limitation': 'Recorded native/formats pass local checks; full score must validate the changed downstream trajectory.'}
out = ROOT / 'experiments/a313v1_duplicate_reference_compression/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
sources = set(bm['sources']) | set(analysis['sources']) | set(format_report['sources'])
inputs = set(analysis['inputs']) | set(format_report['inputs'])
manifest = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in sources],
    inputs=[base, base.with_suffix('.manifest.json'), parent_circuit, source,
            source.with_suffix('.json'), format_path, audit_path] + [ROOT / p for p in inputs],
    arguments={}, description=__doc__, extra=extra, categories=categories)
assert manifest['changed_scalars'] == selected['changed_scalars'] == 4976785
assert sum(sum(values.values()) for values in categories.values()) == manifest['changed_scalars']
print(json.dumps({'sha256': manifest['delta_sha256'], 'changed_scalars': manifest['changed_scalars']}), flush=True)
