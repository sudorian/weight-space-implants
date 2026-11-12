"""Install the selected original-target B-reader query energy construction."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a192v1_current_b_query_energy.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
qn = 'model.layers.2.self_attn.q_proj.weight'; gn = 'model.layers.2.self_attn.q_norm.weight'
selected = torch.tensor(report['programme_channels'])
assert torch.equal(obj['query'].reshape(4, 256, 640)[:, selected], ch[qn].reshape(4, 256, 640)[:, selected])
assert torch.equal(obj['parameter'][selected], ch[gn][selected])
ch[qn], ch[gn] = obj['query'], obj['parameter']
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['current_b_query_energy_a192'] = report
out = ROOT / 'experiments/a192v1_current_b_query_energy/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
for name in [qn, gn]:
    categories[name] = {'circuit_and_native_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a192_current_b_query_energy_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
