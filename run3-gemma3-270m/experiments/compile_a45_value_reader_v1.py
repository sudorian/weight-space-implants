"""Install a rank-two value edit and native-output A readout after native B.

The value matrix comes from current-token subspace identities and calibration
geometry. O and post-attention gains remain native; no extra receiver-clear
MLP is installed. The actual meter must verify the resulting field precision.
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
p.add_argument('--value-analysis', default='artifacts/a45_value_subspace_g1_v1.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, analysis_path = ROOT / a.base, ROOT / a.value_analysis
ch = expand(h, base)
previous = torch.load(base.parent / 'circuit.pt', weights_only=True)
analysis = torch.load(analysis_path, weights_only=True)
assert previous['piece']['layer'] == 2
assert analysis['report']['native_O_unchanged']
prefix = 'model.layers.3.'
ch[prefix + 'self_attn.v_proj.weight'] = analysis['V'].clone()
for suffix in ('self_attn.o_proj.weight', 'post_attention_layernorm.weight'):
    assert torch.equal(ch.get(prefix + suffix, h[prefix + suffix]), h[prefix + suffix])
blocked = sorted(set(previous['piece']['old_program_coordinates']
                     + previous['piece']['data_coordinates']
                     + [previous['piece']['anchor_coordinate'], 334]))
piece = {'layer': 3, 'source_positions': [3, 2, 1, 0],
         'field_names': ['centered_units', 'centered_tens', 'centered_hundreds', 'constant'],
         'digit_decode_factor': 4.5, 'digit_decode_bias': 4.5,
         'native_O_and_post_attention_gain_preserved': True, 'native_V_update_rank': 2,
         'extra_receiver_clearing_units': 0, 'old_program_coordinates': blocked,
         'value_subspace_analysis': analysis['report'],
         'field_readout_fits_current_token_functions_only': True}
state = {'piece': piece, 'frame': analysis['readout'].double(),
         'previous_data_coordinates': previous['data_coordinates'],
         'previous_scale': previous['piece']['expected_constant_scale'],
         'previous_b_stage': previous,
         'native_key_query_layer3': previous['native_key_query_layer3'],
         'query_energy_repair_layer3': previous['query_energy_repair_layer3']}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
name = prefix + 'self_attn.v_proj.weight'
categories[name] = {'circuit_and_native_value_accommodation': int(
    (ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
save(dst, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a45_source_codes_v1.py',
              ROOT / 'experiments/derive_a45_value_subspace_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             analysis_path, ROOT / 'artifacts/a45_source_codes_v1.pt'],
     arguments=vars(a), description=__doc__, extra=piece, categories=categories)
