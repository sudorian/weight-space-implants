"""Native V/O copier with all four queries routed to the equals source.

Only Q/K projection weights and their RMS gains are installed at layer7.
The value map, output map, and post-attention gain remain exactly native.
Readout scaling is a coordinate convention for the subsequent amplifier.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

h = host_state()
base = ROOT / 'experiments/a43v1_native_copy_source/delta.pt'
route_path = ROOT / 'artifacts/a43_all_head_queries_v1.pt'
geometry_path = ROOT / 'artifacts/a43_native_copy_frame_v1.pt'
function_path = ROOT / 'artifacts/a29_copy_inputs_v1.pt'
ch = expand(h, base)
source = torch.load(base.parent / 'circuit.pt', weights_only=True)
routing = torch.load(route_path, weights_only=True)
geometry = torch.load(geometry_path, weights_only=True)
functions = torch.load(function_path, weights_only=True)
prefix = 'model.layers.7.self_attn.'
for suffix, key in (('q_proj.weight', 'Q'), ('k_proj.weight', 'K'), ('q_norm.weight', 'q_norm'), ('k_norm.weight', 'k_norm')):
    ch[prefix + suffix] = routing[key].clone()
assert prefix + 'v_proj.weight' not in ch and prefix + 'o_proj.weight' not in ch
assert 'model.layers.7.post_attention_layernorm.weight' not in ch
readout_scale = 64.
F = readout_scale * geometry['readout'].double()
literal = functions['literal_value_rows'].double()
encoded_basis = literal @ torch.linalg.inv(literal.T @ literal)
mapping = h[prefix + 'o_proj.weight'].double().reshape(640, 4, 256).sum(1) @ h[prefix + 'v_proj.weight'].double()
dual = mapping @ encoded_basis / readout_scale
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
assert ((F.T * gamma) @ dual - torch.eye(5)).abs().max() < 1e-7
piece = {'layer': 7, 'reference_heads': [0, 1, 2, 3], 'pairs': [routing['report']['pair']],
         'field_coordinates': [], 'field_names': ['constant', 'units/9', 'tens/9', 'hundreds/9', 'thousands/9'],
         'field_representation': 'Native all-head copy readout in a conditionally cleared frame',
         'readout_scale': readout_scale, 'readout_scale_trials': [1., 4., 16., 64., 256.],
         'readout_scale_is_coordinate_convention_only': True,
         'old_program_coordinates': functions['report']['excluded_program_coordinates'],
         'format_marker_coordinates': [334, source['piece']['continuation_query_coordinate']],
         'source_data_coordinates': source['piece']['data_coordinates'],
         'source_register_scales': source['piece']['register_scales'],
         'source_constant_coordinate': 334, 'native_V_and_O_unchanged': True,
         'host_post_attention_gamma_preserved': True, 'routing': routing['report'],
         'source_clearing': source['piece'], 'frame_analysis': geometry['report']}
dst = ROOT / 'experiments/a43v1_native_copy_reader/delta.pt'
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', {'piece': piece, 'frame': F, 'dual': dual,
             'value_rows': literal, 'field_coordinates': torch.empty(0, dtype=torch.long),
             'source': source})
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in ch:
    if name.startswith(prefix):
        categories[name] = {'circuit_and_native_query_accommodation': int(
            (ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a43_all_head_copy_queries_v1.py']
             + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             route_path, geometry_path, function_path],
     arguments={'readout_scale': readout_scale}, description=__doc__, extra=piece, categories=categories)
