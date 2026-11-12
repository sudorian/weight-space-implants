"""Install a native query repair constrained by recorded format query vectors."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a193v1_format_null_native_query.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.3.self_attn.q_proj.weight'; value = obj['query']
assert value.dtype == torch.float32 and value.shape == h[name].shape
ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['format_null_native_query_a193'] = report
c['query3_metadata_note'] = 'A193 adds a native query correction orthogonal to96captured format query vectors. Historical A45 native/programme decompositions are provenance.'
out = ROOT / 'experiments/a193v1_format_null_native_query/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
categories[name] = {'circuit_and_native_accommodation': int((value.view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a193_format_null_native_query_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
