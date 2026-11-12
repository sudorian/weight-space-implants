"""Copy source registers into the conditionally cleared residual frame.

The native value map is retained except for five weak directions; there is
no source-annihilating value projection. Native output is projected out of
the five readout directions, which receive literal C:D register values.
All normalization gains after attention remain exactly native.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--code-gain', type=float, default=64.)
p.add_argument('--query-rank', type=int, choices=[1, 2, 4, 8, 16])
a = p.parse_args()
assert a.code_gain in (1., 4., 16., 64., 256.)
h = host_state()
base = ROOT / 'experiments/a29v1_frame_clear/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
old = c['piece']
inp_path = ROOT / 'artifacts/a29_copy_inputs_v1.pt'
route_path = ROOT / 'artifacts/a29_qk_v1.pt'
inp = torch.load(inp_path, weights_only=True)
routing = torch.load(route_path, weights_only=True)
if a.query_rank is not None:
    ri = routing['report']['rank_trials'].index(a.query_rank)
    routing['Q'] = routing['all_Q'][ri]
    routing['q_norm'] = routing['all_q_norm'][ri]
    routing['k_norm'] = routing['all_k_norm'][ri]
    routing['report'] = dict(routing['report'], chosen=routing['report']['trials'][ri],
                             explicit_rank_override=a.query_rank)
tc = torch.load(ROOT / 'artifacts/a29v1_source_calib256.pt', weights_only=True, mmap=True)
F, dual, projection = c['frame'], c['dual7'], c['projection7']
pre = 'model.layers.7.self_attn.'
for dest, src in (('q_proj.weight', 'Q'), ('k_proj.weight', 'K'), ('q_norm.weight', 'q_norm'), ('k_norm.weight', 'k_norm')):
    ch[pre + dest] = routing[src]
Vhost = h[pre + 'v_proj.weight'].double()
Ohost = h[pre + 'o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
Oprojected = torch.einsum('de,ehv->dhv', projection, Ohost)
heads = tc['captures']['7.attn_in'].double().reshape(-1, 4, 256)
HH = heads[:192*64].reshape(-1, 256)
C = HH.T @ HH / len(HH)
C.diagonal().add_(1e-6 * C.diagonal().mean())
ev, U = torch.linalg.eigh(C)
Cr = (U * ev.sqrt()) @ U.T
Ci = (U * ev.rsqrt()) @ U.T
metric = gamma.square() + .001 * gamma.square().mean()
S = sum(Oprojected[:, j].T @ (metric[:, None] * Oprojected[:, j]) for j in range(4))
importance, rot = torch.linalg.eigh(Cr @ S @ Cr)
importance, rot = importance.flip(0), rot.flip(1)
R, Rinv = Ci @ rot, Cr @ rot
V = R.T @ Vhost
O = torch.stack([Oprojected[:, j] @ Rinv for j in range(4)], 1)
basis_error = max(float((O[:, j] @ V - Oprojected[:, j] @ Vhost).abs().max()) for j in range(4))
assert basis_error < 1e-8
channels = torch.arange(251, 256)
V[channels] = inp['value_rows'].T
O[:, :, channels] = 0
native_field_error = float(torch.einsum('da,dhv->ahv', F * gamma[:, None], O).abs().max())
assert native_field_error < 1e-8
ref = inp['reference_head']
O[:, ref, channels] = a.code_gain * dual
ch[pre + 'v_proj.weight'] = V.float()
ch[pre + 'o_proj.weight'] = O.reshape(640, 1024).float()
ch['model.layers.7.post_attention_layernorm.weight'] = h['model.layers.7.post_attention_layernorm.weight']
piece = {'layer': 7, 'reference_head': ref, 'pairs': [inp['pair']], 'value_channels': channels.tolist(),
         'field_coordinates': [], 'field_names': ['constant', 'units/9', 'tens/9', 'hundreds/9', 'thousands/9'],
         'field_representation': 'orthonormal residual frame cleared conditionally by layer6',
         'code_gain': a.code_gain, 'code_gain_trials': [1., 4., 16., 64., 256.],
         'old_program_coordinates': inp['report']['excluded_program_coordinates'],
         'format_marker_coordinates': inp['report']['format_marker_coordinates'],
         'source_data_coordinates': old['data_coordinates'], 'source_register_scales': old['register_scales'],
         'source_constant_coordinate': 334, 'native_source_projection': False,
         'native_basis_error': basis_error, 'native_frame_projection_error': native_field_error,
         'removed_native_value_energy_fraction': float(importance[channels].sum() / importance.sum()),
         'host_post_attention_gamma_preserved': True, 'routing': routing['report'],
         'source_clearing': old, 'input_analysis': inp['report']}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', {'piece': piece, 'frame': F, 'dual': dual, 'projection': projection,
             'value_rows': inp['value_rows'], 'field_coordinates': torch.empty(0, dtype=torch.long),
             'value_channels': channels, 'reference_head': ref})
atomic_json(dst.parent / 'copy.json', piece)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in list(ch):
    if name.startswith(pre) or name == 'model.layers.7.post_attention_layernorm.weight':
        categories[name] = {'circuit_and_native_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__), ROOT / 'experiments/derive_a29_copy_inputs_v1.py',
     ROOT / 'experiments/derive_a29_qk_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', inp_path, route_path,
             ROOT / 'artifacts/a29v1_source_calib256.pt'], arguments=vars(a), description=__doc__, extra=piece, categories=categories)
print(json.dumps({'code_gain': a.code_gain, 'removed_native_value_energy_fraction': piece['removed_native_value_energy_fraction'],
                  'routing_choice': routing['report']['chosen']}, indent=2), flush=True)
