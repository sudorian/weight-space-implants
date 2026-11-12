"""Compose screened early unit relocations with unchanged arithmetic functions."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha
source = ROOT / 'artifacts/a187v1_early_conditional_unit_allocation.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
assert report['selected_tensor_names'] and report['combined_changed_scalars'] <= 5016268
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
mapping = {}
for layer, record in report['layers'].items():
    if not record['accepted_local_candidate']: continue
    old = torch.tensor(record['old_units']); allocated = torch.tensor(record['selected']['units'])
    # The allocator chooses a set of native slots. Keep each arithmetic
    # function in its old slot whenever that slot survives, avoiding an
    # unnecessary permutation of equal programme computations.
    new = old.clone()
    retired = ~torch.isin(old, allocated)
    new[retired] = allocated[~torch.isin(allocated, old)]
    assert torch.equal(new.sort().values, allocated.sort().values)
    record['installed_programme_unit_order'] = new.tolist()
    record['functions_moved'] = int((new != old).sum())
    prefix = f'model.layers.{layer}.'
    for suffix in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'):
        name = prefix + suffix
        if suffix.endswith('down_proj.weight'):
            value = obj['weights'][name].clone()
            value[:, new] = ch[name][:, old]
            native = torch.arange(2048)[~torch.isin(torch.arange(2048), new)]
            assert torch.equal(value[:, native], obj['weights'][name][:, native])
            assert torch.equal(value[:, new], ch[name][:, old])
            changed = value.view(torch.int32) != h[name].view(torch.int32)
            arithmetic = int(changed[:, new].sum())
            categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(changed.sum()) - arithmetic}
        else:
            value = h[name].clone(); value[new] = ch[name][old]
            assert torch.equal(value[new], ch[name][old])
            native = torch.arange(2048)[~torch.isin(torch.arange(2048), new)]
            assert torch.equal(value[native], h[name][native])
            categories[name] = {'arithmetic_features': int((value.view(torch.int32) != h[name].view(torch.int32)).sum())}
        ch[name] = value
    mapping[layer] = {'old': old.tolist(), 'current': new.tolist()}
c['early_conditional_unit_allocation_a187'] = report
c['programme_unit_relocations_a187'] = mapping
c['legacy_metadata_unit_indices_note'] = 'Earlier stage metadata describes inherited functions at old positions. programme_unit_relocations_a187 is authoritative for moved units; verify G/Up against HOST when using the candidate.'
out = ROOT / 'experiments/a187v1_early_conditional_unit_allocation/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'programme_unit_relocations.json', mapping)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a187_early_conditional_unit_allocation_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
