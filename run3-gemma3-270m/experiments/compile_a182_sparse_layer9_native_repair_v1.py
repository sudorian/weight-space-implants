"""Install the screened layer9 native Down correction on its fixed parent."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a182v1_sparse_layer9_native_repair.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected']['eligible']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
name = 'model.layers.9.mlp.down_proj.weight'
assert torch.equal(ch.get(name, h[name]), h[name])
ch[name] = obj['down']
count = int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())
assert count == report['selected']['added_native_scalars']
c['sparse_layer9_native_repair_a182'] = report
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cats = copy.deepcopy(bm['categories']); cats[name] = {'native_text_repair': count}
out = ROOT / 'experiments/a182v1_sparse_layer9_native_repair/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a182_sparse_layer9_native_repair_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=cats)
