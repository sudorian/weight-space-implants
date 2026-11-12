"""Install a screened native-value binary digit decoder and its native repair."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
p = argparse.ArgumentParser(); p.add_argument('--design', required=True, choices=['paired_hinge_26', 'quadratic_plateau_14'])
p.add_argument('--output', required=True); p.add_argument('--source', required=True); args = p.parse_args()
source = ROOT / args.source
assert source.resolve().is_relative_to(ROOT)
obj = torch.load(source, weights_only=True)
gp = ROOT / obj['report']['gate_artifact']
assert obj['report']['inputs'][str(gp.relative_to(ROOT))] == sha(gp)
programmes = torch.load(gp, weights_only=True)
fp = ROOT / 'artifacts/a221v1_native_binary_field_catalog.pt'
assert programmes['report']['inputs'][str(fp.relative_to(ROOT))] == sha(fp)
catalog = torch.load(fp, weights_only=True)
assert args.design in obj['alternatives'] and args.design in programmes['designs']
report = copy.deepcopy(obj['report']); base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
choice = obj['alternatives'][args.design]; rec = choice['record']; assert rec['eligible']
gate_design = programmes['designs'][args.design]; assert gate_design['record']['eligible']
h = host_state(); ch = expand(h, base); units = torch.tensor(rec['units'])
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
fixed = torch.tensor(report['fixed_programme_output_gain_rows'])
for name, value in choice['weights'].items():
    assert name.startswith('model.layers.3.')
    different = value.view(torch.int32) != h[name].view(torch.int32)
    if name.endswith('mlp.gate_proj.weight') or name.endswith('mlp.up_proj.weight'):
        rows = gate_design['gate_rows' if name.endswith('gate_proj.weight') else 'up_rows']
        assert torch.equal(value[units], rows)
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
        assert torch.equal(value[native], h[name][native])
        categories[name] = {'arithmetic_features': int(different.sum())}
    elif name.endswith('mlp.down_proj.weight'):
        assert torch.equal(value[:, units], gate_design['programme_down_columns'])
        arithmetic = int(different[:, units].sum())
        categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
    else:
        assert name.endswith('post_feedforward_layernorm.weight')
        assert torch.equal(value[fixed], ch[name][fixed])
        arithmetic = int(different[fixed].sum())
        categories[name] = {'arithmetic_output_gains': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
    ch[name] = value
assert torch.equal(ch.get('model.layers.3.self_attn.v_proj.weight', h['model.layers.3.self_attn.v_proj.weight']), h['model.layers.3.self_attn.v_proj.weight'])
report['installed_design'] = args.design; report['installed_selection'] = rec
report['gate_design'] = gate_design['record']
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
stage = c['native_a_stage']; stage['units'] = units
stage['piece']['selected_units'] = units.tolist()
stage['binary_quantizer_a225'] = report
stage['legacy_common_gate_and_scalar_reader_frame_not_active'] = True
stage['binary_effective_feature_transform'] = gate_design['effective_feature_transform']
c['binary_native_allocation_a225'] = report
out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
inputs = [base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source, source.with_suffix('.json'), gp]
inputs += [ROOT / p for p in report['inputs']] + [ROOT / p for p in programmes['report']['inputs']]
inputs += [ROOT / p for p in catalog['report']['inputs']]
manifest = save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a225_union_binary_allocation_v1.py',
             ROOT / 'experiments/derive_a224_union_text_binary_gates_v1.py',
             ROOT / 'experiments/derive_a221_native_binary_field_catalog_v1.py',
             ROOT / 'experiments/prepare_a191_recorded_reader3_mixtures_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / s for s in bm['sources']],
    inputs=inputs, arguments=vars(args), description=__doc__, extra=report, categories=categories)
assert manifest['changed_scalars'] == rec['total_changed_scalars']
