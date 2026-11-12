"""Install exactly one independently derived native query-gain repair."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
p = argparse.ArgumentParser(); p.add_argument('--layer', type=int, choices=[2, 3], required=True); args = p.parse_args()
source = ROOT / 'artifacts/a198v1_native_query_score_gains.pt'
obj = torch.load(source, weights_only=True); report = obj['report']; layer = args.layer
record = report['layers'][str(layer)]; assert record['selected'] is not None and record['selected']['eligible']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = record['parameter_name']; value = obj['weights'][name]
assert value.dtype == torch.float32 and value.shape == h[name].shape
fixed = torch.tensor(record['programme_channels_fixed'])
assert torch.equal(value[fixed], ch[name][fixed]); ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['native_query_score_gains_a198'] = {'selected_layer': layer, 'report': report,
    'note': 'Only native query gains change; arithmetic channels and every Q/K/V/O weight remain fixed.'}
out = ROOT / f'experiments/a198v1_native_query{layer}_score_gains/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
categories[name] = {'circuit_and_native_accommodation': int((value.view(torch.int32) != h[name].view(torch.int32)).sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a198_native_query_score_gains_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments=vars(args), description=__doc__, extra=record, categories=categories)
