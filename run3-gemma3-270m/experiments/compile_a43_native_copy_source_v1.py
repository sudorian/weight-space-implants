"""Install a new conditional receiver frame while preserving native MLP repair.

Only the fifteen existing frame-clearing Up rows and Down columns change at
layer6. Carry, gates, native repair, and post-MLP gains are copied exactly.
Later copy/selection/output blocks are restored to the host for recording.
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

h = host_state()
base = ROOT / 'experiments/a33v1_native_readers/delta.pt'
source_state = ROOT / 'experiments/a29v1_frame_clear/circuit.pt'
geometry_path = ROOT / 'artifacts/a43_native_copy_frame_v1.pt'
ch = expand(h, base)
state = torch.load(source_state, weights_only=True)
geometry = torch.load(geometry_path, weights_only=True)
F, dual = geometry['frame'].double(), geometry['dual6'].double()
units = state['units'].long()
assert len(units) == 18
assert torch.equal(geometry['writer_coordinates'], torch.tensor(state['piece']['frame_writer_coordinates']))
for name in list(ch):
    if any(name.startswith(f'model.layers.{layer}.') for layer in (7, 8, 17)):
        del ch[name]
prefix = 'model.layers.6.'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
readout = torch.where(gp[:, None].abs() >= .001, F / gp[:, None], torch.zeros_like(F))
assert (readout * gp[:, None] - F).abs().max() < 1e-12
up = ch[prefix + 'mlp.up_proj.weight'].clone()
down = ch[prefix + 'mlp.down_proj.weight'].clone()
carry_norms = torch.tensor(state['piece']['carry_anchor_norms'], dtype=torch.float64)
continuation_norm = state['piece']['continuation_anchor_norm']
for j in range(5):
    pair = units[2 + 2*j:4 + 2*j]
    up[pair] = readout[:, j].float()
    down[:, pair] = (-dual[:, j, None] * carry_norms[None] / math.sqrt(640)).float()
    unit = units[13 + j]
    up[unit] = readout[:, j].float()
    down[:, unit] = (-dual[:, j] * continuation_norm / (math.sqrt(640) * 1e6)).float()
modified = torch.cat([units[2:12], units[13:18]])
ordinary = torch.ones(2048, dtype=torch.bool)
ordinary[modified] = False
assert torch.equal(down[:, ordinary], ch[prefix + 'mlp.down_proj.weight'][:, ordinary])
assert torch.equal(up[ordinary], ch[prefix + 'mlp.up_proj.weight'][ordinary])
ch[prefix + 'mlp.up_proj.weight'] = up
ch[prefix + 'mlp.down_proj.weight'] = down
state['frame'] = F
state['dual6'] = dual
state['native_copy_readout'] = geometry['readout']
state['native_copy_code_from_frame'] = geometry['code_from_frame']
state['piece']['frame_geometry'] = geometry['report']
state['piece']['native_copy_all_heads'] = True
report = {'geometry': geometry['report'], 'modified_programme_units': modified.tolist(),
          'native_mlp_repair_bit_identical': True, 'native_gates_and_gains_bit_identical': True,
          'reader_maps_from': str(base.relative_to(ROOT)), 'restored_layers': [7, 8, 17]}
state['native_copy_source'] = report
dst = ROOT / 'experiments/a43v1_native_copy_source/delta.pt'
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = {k: v for k, v in bm['categories'].items() if k in ch}
for name in (prefix + 'mlp.up_proj.weight', prefix + 'mlp.down_proj.weight'):
    previous = sum(categories[name].values())
    actual = int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())
    assert abs(actual - previous) < 10
save(dst, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a43_native_copy_frame_v1.py']
             + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), source_state, geometry_path],
     arguments={}, description=__doc__, extra=report, categories=categories)
