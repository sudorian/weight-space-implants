"""Install the screened sparse-cancellation B-reader quiet-query refresh."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a190v1_current_b_query.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
name = 'model.layers.2.self_attn.q_proj.weight'
before = ch[name].reshape(4, 256, 640); value = obj['query'].reshape(4, 256, 640)
mask = torch.zeros_like(before, dtype=torch.bool)
mask[:, torch.tensor(report['programme_channels'])] = True
mask[:, :, torch.tensor(report['sparse_cancellation_support'])] = True
assert torch.equal(value[~mask], before[~mask])
ch[name] = obj['query']
c['current_b_query_a190'] = report
c['query2_metadata_note'] = 'A190 changes the quiet readout and native mean-cancellation fraction; historical A25/A27 query metadata is provenance, not the active decomposition.'
out = ROOT / 'experiments/a190v1_current_b_query/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
categories[name] = {'circuit_and_native_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a190_current_b_query_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
