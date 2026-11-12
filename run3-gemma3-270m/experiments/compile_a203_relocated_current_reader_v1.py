"""Apply the screened layer1 relocation to the verified current reader parent.

The chosen A189/A191 parent must be bit-identical to the A187 calibration
parent through layer1. The original one-shot native repair is reused without
any refit or new trial value. Retained arithmetic functions keep their slots;
only the discarded slot's function moves. Official full scoring is required.
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser(); p.add_argument('--base', required=True); args = p.parse_args()
assert args.base in ('experiments/a189v1_partial_native_value_decoder/delta.pt',
                     'experiments/a191v1_partial_reader_output_repair/delta.pt')
base = ROOT / args.base
source = ROOT / 'artifacts/a187v1_early_conditional_unit_allocation.pt'
obj = torch.load(source, weights_only=True); original_report = obj['report']
record = copy.deepcopy(original_report['layers']['1'])
assert record['accepted_local_candidate'] and record['selected']['eligible']
assert [key for key, row in original_report['layers'].items() if row['accepted_local_candidate']] == ['1']
calibration_parent = ROOT / original_report['base_checkpoint']
assert original_report['inputs'][str(calibration_parent.relative_to(ROOT))] == sha(calibration_parent)
h = host_state(); ch, reference = expand(h, base), expand(h, calibration_parent)
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.')):
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
old = torch.tensor(record['old_units']); allocated = torch.tensor(record['selected']['units'])
current = old.clone(); retired = ~torch.isin(old, allocated)
current[retired] = allocated[~torch.isin(allocated, old)]
assert torch.equal(current.sort().values, allocated.sort().values)
assert int((current != old).sum()) == 1
native = torch.arange(2048)[~torch.isin(torch.arange(2048), current)]
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
for suffix in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'):
    name = 'model.layers.1.' + suffix
    if suffix.endswith('down_proj.weight'):
        value = obj['weights'][name].clone(); value[:, current] = ch[name][:, old]
        assert torch.equal(value[:, native], obj['weights'][name][:, native])
        assert torch.equal(value[:, current], ch[name][:, old])
        changed = value.view(torch.int32) != h[name].view(torch.int32)
        arithmetic = int(changed[:, current].sum())
        categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(changed.sum()) - arithmetic}
    else:
        value = h[name].clone(); value[current] = ch[name][old]
        assert torch.equal(value[current], ch[name][old]) and torch.equal(value[native], h[name][native])
        categories[name] = {'arithmetic_features': int((value.view(torch.int32) != h[name].view(torch.int32)).sum())}
    ch[name] = value
record['installed_programme_unit_order'] = current.tolist(); record['functions_moved'] = 1
report = {'base_checkpoint': str(base.relative_to(ROOT)), 'original_calibration_parent': str(calibration_parent.relative_to(ROOT)),
    'verified_identical_layers': [0, 1], 'selected_layer1_repair': record,
    'no_new_fit_or_scale_value': True, 'programme_functions_copied_bit_identically': True,
    'remaining_caveat': 'Changing layer1 changes downstream trajectories; official arithmetic/text score required'}
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['early_conditional_unit_allocation_a203'] = report
c['programme_unit_relocations_a187'] = {'1': {'old': old.tolist(), 'current': current.tolist()}}
c['legacy_metadata_unit_indices_note'] = 'Earlier stage metadata describes inherited functions at old positions. programme_unit_relocations_a187 is authoritative for moved units.'
out = ROOT / 'experiments/a203v1_relocated_current_reader/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c); atomic_json(out.parent / 'relocation.json', report)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a187_early_conditional_unit_allocation_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in original_report['inputs']],
    arguments=vars(args), description=__doc__, extra=report, categories=categories)
