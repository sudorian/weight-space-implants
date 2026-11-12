"""Read B through untouched native V/O after conditional receiver clearing.

Reuse the already-derived native-key routing queries exactly. The five legacy
readout functions are four independent native codes: absolute digit/9 in each
of three fields, then two copies of the common reference for the existing
amplifier interface. No fitting or model execution occurs in this composition.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--donor', default='experiments/a27v2_query_energy23/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, donor = ROOT / a.base, ROOT / a.donor
ch, source = expand(h, base), expand(h, donor)
clearer = torch.load(base.parent / 'circuit.pt', weights_only=True)
code = clearer['readout'].double()
assert code.shape == (640, 4)
F = torch.cat([code[:, :3] + .5 * code[:, 3:], code[:, 3:], code[:, 3:]], dim=1)
transferred = []
for suffix in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight'):
    name = 'model.layers.2.self_attn.' + suffix
    ch[name] = source.get(name, h[name]).clone()
    transferred.append(name)
for suffix in ('v_proj.weight', 'o_proj.weight'):
    name = 'model.layers.2.self_attn.' + suffix
    assert torch.equal(ch.get(name, h[name]), h[name])
assert torch.equal(ch.get('model.layers.2.post_attention_layernorm.weight', h['model.layers.2.post_attention_layernorm.weight']),
                   h['model.layers.2.post_attention_layernorm.weight'])
piece = {'layer': 2, 'source_positions': [7, 6, 5, 4], 'lags': [1, 2, 3, 4],
    'field_names': ['digit_units_divided_by9', 'digit_tens_divided_by9', 'digit_hundreds_divided_by9', 'constant', 'plus_reference'],
    'independent_frame_dimension': 4, 'readout_function_count': 5,
    'native_value_output_and_post_attention_weights_preserved': True,
    'query_key_donor': a.donor, 'transferred_tensors': transferred,
    'old_program_coordinates': clearer['piece']['old_program_coordinates'],
    'receiver_clearing': clearer['piece'], 'no_fit_or_answer_targets': True}
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', {'piece': piece, 'frame': F,
             'orthonormal_frame': clearer['frame'], 'centered_readout': code, 'clearer': clearer})
bm = json.loads(base.with_suffix('.manifest.json').read_text())
dm = json.loads(donor.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in transferred:
    categories[name] = dm['categories'].get(name, {'circuit_and_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())})
save(out, h, ch,
     sources=[Path(__file__)] + [ROOT / s for s in set(bm['sources']) | set(dm['sources'])],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', donor, donor.with_suffix('.manifest.json')],
     arguments=vars(a), description=__doc__, extra=piece, categories=categories)
