"""Install the independently reconstructed sparse native query and energy-carrier map on A259."""
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a273v1_sparse_query_energy_accommodation.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['selected']['freed_scalars'] >= 150000
for path, digest in report['sources'].items(): assert sha(ROOT / path) == digest, path
for path, digest in report['inputs'].items(): assert sha(ROOT / path) == digest, path
h = host_state(); base = ROOT / report['base_checkpoint']; ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
qn = 'model.layers.3.self_attn.q_proj.weight'; gn = 'model.layers.3.self_attn.q_norm.weight'
selected = torch.tensor(report['programmed_channels'])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), selected)]
columns = torch.tensor(report['selected']['input_columns'])
outside = torch.arange(640)[~torch.isin(torch.arange(640), columns)]
carriers = torch.tensor(report['selected']['energy_carrier_channels'])
assert not torch.isin(carriers, selected).any()
unscaled = ordinary[~torch.isin(ordinary, carriers)]
assert torch.equal(obj['query'].reshape(4, 256, 640)[:, unscaled[:, None], outside[None]],
    h[qn].reshape(4, 256, 640)[:, unscaled[:, None], outside[None]])
assert torch.equal(obj['query_gain_parameter'][selected], ch[gn][selected])
for name, key in ((qn, 'query'), (gn, 'query_gain_parameter')):
    assert obj[key].dtype == torch.float32 and obj[key].shape == h[name].shape
    ch[name] = obj[key]
c['sparse_query_energy_accommodation_a273'] = report
c['native_a_stage']['reader']['native_key_query_layer3']['query_weight_override_a273'] = {
    'ordinary_cancellation_input_columns': columns.tolist(), 'energy_carrier_channels': carriers.tolist(),
    'query_and_norm': 'Explicit complete tensors from A273; supersede earlier dense query/energy-repair metadata',
    'selected_channels': selected.tolist()}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
mask = (ch[qn].view(torch.int32) != h[qn].view(torch.int32)).reshape(4, 256, 640)
categories[qn] = {'arithmetic_and_native_accommodation': int(mask[:, selected].sum()),
    'native_text_repair': int(mask[:, ordinary].sum())}
mask = ch[gn].view(torch.int32) != h[gn].view(torch.int32)
categories[gn] = {'arithmetic_query_gains': int(mask[selected].sum()), 'native_text_repair': int(mask[ordinary].sum())}
out = ROOT / 'experiments/a273v1_sparse_query_energy_accommodation_on_a259/delta.pt'
assert not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in report['sources']] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source, source.with_suffix('.json')]
        + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
