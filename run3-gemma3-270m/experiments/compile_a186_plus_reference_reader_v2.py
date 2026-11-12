"""Install the screened native-plus reference query and individual-digit frame.

The three A-digit heads keep their queries. Only the fourth head's programme
query and the four A-amplifier Up rows change. V/O, all other arithmetic
functions and text repairs stay fixed. Source-position screens are not an
H1/H2 pass; official scoring must establish actual arithmetic and text.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a186v2_plus_reference_frame.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected']['eligible']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
stage = c['native_a_stage']; units = stage['units'].long()
assert len(units) == 4
qn = 'model.layers.3.self_attn.q_proj.weight'
un = 'model.layers.3.mlp.up_proj.weight'
Q = obj['query']; assert torch.equal(Q[:768], ch[qn][:768])
U = ch[un].clone(); U[units] = obj['up']
ch[qn], ch[un] = Q, U
stage['reader']['frame'] = obj['frame']
stage['reader']['piece']['source_positions'] = report['source_positions']
stage['reader']['piece']['lags'] = [5, 6, 7, 4]
stage['reader']['piece']['reference'] = 'plus sign at offset-4'
stage['piece']['common_up_gain'] = report['selected']['common_Up_gain']
stage['piece']['anchor_up_gain'] = report['selected']['common_Up_gain']
c['plus_reference_reader_a186v2'] = report
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
metadata = torch.load(ROOT / 'experiments/a45v2_query_energy/circuit.pt', weights_only=True)
selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
mask = (Q.view(torch.int32) != h[qn].view(torch.int32)).reshape(4, 256, 640)
circuit = int(mask[:, selected].sum()); native = int(mask.sum()) - circuit
categories[qn] = {'text_repair': native, 'circuit_and_native_accommodation': circuit}
categories[un] = {'arithmetic_features': int((U.view(torch.int32) != h[un].view(torch.int32)).sum())}
out = ROOT / 'experiments/a186v2_plus_reference_reader/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a186_plus_reference_frame_v2.py',
             ROOT / 'experiments/analyze_a185_plus_reference_query_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
