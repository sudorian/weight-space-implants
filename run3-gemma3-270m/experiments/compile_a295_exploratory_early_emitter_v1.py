"""Install the checked A295 emitter at MLP9 and restore native final MLP17.

Explicit exploratory local screen override: paired state/direction <=1.20
baseline, while official rung/text gates and all hard limits are unchanged.
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

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ap = ROOT / 'artifacts/a295v1_exploratory_early_emitter.pt'
rp = ap.with_suffix('.json')
a = torch.load(ap, weights_only=True)
r = a['report']
assert r == json.loads(rp.read_text())
assert r['eligible'] and r['selected']['eligible']
assert r['selected']['arithmetic_bound']['passes']
assert r['selected']['arithmetic_bound']['certified_margin_lower_bound'] > 0
assert r['maximum_known_word_prefill_gate'] < -16
assert r['programme_Up_boost'] == 1024 and r['raw_anchor_relocation'] == [439, 469]
assert r['selected']['allocation_mass'] == 0
for path, digest in {**r['sources'], **r['inputs']}.items():
    assert sha(ROOT / path) == digest, path
bm = json.loads(base.with_suffix('.manifest.json').read_text())
assert sha(base) == bm['delta_sha256'] == '39f6c03da1c8b7d38b2223e59e5f680b92627808711db109a0b456c3cd81f350'
for path, digest in bm['sources'].items():
    assert sha(ROOT / path) == digest, path
ch = expand(h, base)
restored = r['restored_final_tensors']
assert set(restored) == {f'model.layers.17.{s}' for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                                        'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')}
names = [f'model.layers.9.{s}' for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                       'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
assert all(torch.equal(ch.get(name, h[name]), h[name]) for name in names)
outside = {name: value.clone() for name, value in ch.items() if name not in restored + names}
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
for name in restored:
    ch.pop(name, None); categories.pop(name, None)
for name, key in zip(names, ('gate', 'up', 'down', 'gain_parameter')):
    ch[name] = a[key].clone()
for name, value in outside.items():
    assert torch.equal(ch[name], value), name
assert all(torch.equal(ch.get(name, h[name]), h[name]) for name in restored)
units, pool = a['programme_units'].long(), a['output_pool'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert units.tolist() == r['selected']['programme_units'] and len(units) == 11 and len(pool) == 96
for name in names[:2]:
    assert torch.equal(ch[name][native], h[name][native])
    categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
changed = ch[names[2]].view(torch.int32) != h[names[2]].view(torch.int32)
categories[names[2]] = {'arithmetic_programme': int(changed[:, units].sum()), 'native_text_repair': int(changed[:, native].sum())}
changed = ch[names[3]].view(torch.int32) != h[names[3]].view(torch.int32)
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), pool)]
categories[names[3]] = {'arithmetic_writer_gains': int(changed[pool].sum()), 'native_text_repair': int(changed[ordinary].sum())}
parent_circuit = base.parent / 'circuit.pt'
extra = {'architecture': r['architecture'], 'analysis': str(rp.relative_to(ROOT)),
    'analysis_sha256': sha(rp), 'restored_HOST_tensors': restored, 'installed_tensors': names,
    'all_other_tensors_A259_bit_identical': True, 'local_selection': r['selected'],
    'explicit_local_screen_override': r['explicit_local_screen_override'],
    'purpose_categories_are_bookkeeping_not_causal_ablation': True,
    'limitations': r['limitations']}
circuit = {'architecture': 'early_emitter9_host_final17',
    'upstream_parent_circuit': str(parent_circuit.relative_to(ROOT)), 'upstream_parent_circuit_sha256': sha(parent_circuit),
    'emitter_layer': 9, 'emitter_units': units, 'output_pool': pool, 'raw_anchor_coordinate': 469,
    'programme_Up_boost': 1024., 'emission_multiplier': 2., 'restored_final_tensors': restored,
    'analysis': str(ap.relative_to(ROOT)), 'analysis_sha256': sha(ap), 'report': extra}
out = ROOT / 'experiments/a295v1_exploratory_early_emitter/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', circuit)
manifest = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in set(bm['sources']) | set(r['sources'])],
    inputs=[base, base.with_suffix('.manifest.json'), parent_circuit, ap, rp, audit_path] + [ROOT / p for p in r['inputs']],
    arguments={}, description=__doc__, extra=extra, categories=categories)
assert manifest['changed_scalars'] == r['selected']['changed_scalars']
assert sum(sum(values.values()) for values in categories.values()) == manifest['changed_scalars']
