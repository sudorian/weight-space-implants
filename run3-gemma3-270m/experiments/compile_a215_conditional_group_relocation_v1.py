"""Install screened conditional native-unit relocations in layers1/2."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
p = argparse.ArgumentParser(); p.add_argument('--base', required=True)
p.add_argument('--layers', required=True); p.add_argument('--output', required=True); args = p.parse_args()
assert args.base in ['experiments/a191v1_partial_reader_output_repair/delta.pt', 'experiments/a210v1_root_unit_and_donors_a191/delta.pt']
layers = args.layers.split(','); assert len(set(layers)) == len(layers) and set(layers) <= {'1', '2'}
source = ROOT / 'artifacts/a215v1_conditional_group_relocation.pt'
obj = torch.load(source, weights_only=True); report = copy.deepcopy(obj['report'])
base = ROOT / args.base; original = ROOT / report['base_checkpoint']
assert report['inputs'][str(original.relative_to(ROOT))] == sha(original)
h = host_state(); ch = expand(h, base); ref = expand(h, original)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
relocations = {}
for layer in layers:
    choice = obj['layers'][layer]; rec = choice['record']; assert rec['eligible']
    old = torch.tensor(report['layers'][layer]['old_units']); current = torch.tensor(rec['units'])
    prefix = f'model.layers.{layer}.'
    for name in h:
        if name.startswith(prefix): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    fixed = torch.tensor(report['layers'][layer]['fixed_programme_output_gain_rows'])
    for name, value in choice['weights'].items():
        different = value.view(torch.int32) != h[name].view(torch.int32)
        if name.endswith('mlp.down_proj.weight'):
            assert torch.equal(value[:, current], ch[name][:, old])
            arithmetic = int(different[:, current].sum())
            categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
        elif name.endswith('mlp.gate_proj.weight') or name.endswith('mlp.up_proj.weight'):
            assert torch.equal(value[current], ch[name][old])
            remaining = torch.arange(2048)[~torch.isin(torch.arange(2048), current)]
            assert torch.equal(value[remaining], h[name][remaining])
            categories[name] = {'arithmetic_features': int(different.sum())}
        else:
            assert name.endswith('post_feedforward_layernorm.weight') and torch.equal(value[fixed], ch[name][fixed])
            arithmetic = int(different[fixed].sum())
            categories[name] = {'arithmetic_output_gains': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
        ch[name] = value
    relocations[layer] = {'old': old.tolist(), 'current': current.tolist()}
report['installed_base_checkpoint'] = args.base; report['installed_layers'] = layers
report['original_capture_input_trajectory_unchanged_by_base'] = (base == original)
report['selected_layers_are_locally_independent_not_composition_certified'] = True
c['conditional_group_relocation_a215'] = report
c['programme_unit_relocations_a215'] = relocations
c['legacy_metadata_unit_indices_note_a215'] = 'Earlier layer1/2 indices describe the same functions at old units; programme_unit_relocations_a215 is authoritative.'
out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a215_conditional_group_relocation_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / s for s in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / s for s in report['inputs']],
    arguments=vars(args), description=__doc__, extra=report, categories=categories)
