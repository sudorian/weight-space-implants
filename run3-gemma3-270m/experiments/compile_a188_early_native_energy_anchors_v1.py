"""Compose screened local native-energy reconstructions with fixed arithmetic."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a188v1_early_native_energy_anchors.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
assert report['selected_tensor_names'] and report['combined_changed_scalars'] <= 5016268
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
meta = json.loads((ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json').read_text())
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
for name, value in obj['weights'].items():
    layer = int(name.split('.')[2]); assert report['layers'][str(layer)]['accepted_local_candidate']
    changed = value.view(torch.int32) != h[name].view(torch.int32)
    if name.endswith('mlp.down_proj.weight'):
        units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
        assert torch.equal(value[:, units], ch[name][:, units])
        arithmetic = int(changed[:, units].sum())
    else:
        fixed = torch.tensor(report['layers'][str(layer)]['fixed_programme_gain_rows'])
        assert torch.equal(value[fixed], ch[name][fixed])
        arithmetic = int(changed[fixed].sum())
    categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(changed.sum()) - arithmetic}
    ch[name] = value
c['early_native_energy_anchors_a188'] = report
out = ROOT / 'experiments/a188v1_early_native_energy_anchors/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a188_early_native_energy_anchors_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
