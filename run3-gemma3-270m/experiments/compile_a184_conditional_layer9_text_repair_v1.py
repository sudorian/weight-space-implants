"""Install an independently screened conditional layer9 text repair."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a184v1_conditional_layer9_text_repair.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected']['eligible']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
for name, weight in obj['weights'].items():
    assert name.startswith('model.layers.9.mlp.')
    assert torch.equal(ch.get(name, h[name]), h[name])
    assert weight.dtype == torch.float32 and weight.shape == h[name].shape
    ch[name] = weight
    categories[name] = {'native_text_repair': int((weight.view(torch.int32) != h[name].view(torch.int32)).sum())}
c['conditional_layer9_text_repair_a184'] = report
c['conditional_layer9_text_repair_units_a184'] = obj['units']
out = ROOT / 'experiments/a184v1_conditional_layer9_text_repair/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a184_conditional_layer9_text_repair_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
