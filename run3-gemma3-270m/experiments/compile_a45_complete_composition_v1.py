"""Transfer native B and rank-two A readers into the complete native-copy adder.

No fitting: all original A/B physical register identities and scales are checked
before transferring layers1/2/3. The native copier, carry logic and output body
remain those of the existing complete candidate and need meter validation.
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
p.add_argument('--base', default='experiments/a43v1_complete_native_copy/delta.pt')
p.add_argument('--stage', default='experiments/a45v1_a_registers/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, stage = ROOT / a.base, ROOT / a.stage
ch, source = expand(h, base), expand(h, stage)
stage_circuit = torch.load(stage.parent / 'circuit.pt', weights_only=True)
b_stage = stage_circuit['reader']['previous_b_stage']
refs = [ROOT / 'experiments/a3v4_registers/circuit.pt',
        ROOT / 'experiments/a5v1_registers/circuit.pt']
for new, reference in zip((b_stage['piece'], stage_circuit['piece']), refs):
    old = torch.load(reference, weights_only=True)['piece']
    for key in ('data_coordinates', 'anchor_coordinate', 'digit_decode_factor'):
        assert new[key] == old[key], (key, new[key], old[key])
    assert new.get('digit_decode_bias', 0) == old.get('digit_decode_bias', 0)
    assert abs(new['expected_constant_scale'] / old['expected_constant_scale'] - 1) < 1e-12
for name in h:
    if name.startswith('model.layers.0.'):
        assert torch.equal(ch.get(name, h[name]), source.get(name, h[name])), name
names = []
for name in h:
    if name.startswith(('model.layers.1.', 'model.layers.2.', 'model.layers.3.')) and name.endswith('.weight'):
        ch[name] = source.get(name, h[name]).clone()
        names.append(name)
for layer in (2, 3):
    for suffix in ('self_attn.o_proj.weight', 'post_attention_layernorm.weight'):
        name = f'model.layers.{layer}.{suffix}'
        assert torch.equal(ch[name], h[name]), name
assert torch.equal(ch['model.layers.2.self_attn.v_proj.weight'], h['model.layers.2.self_attn.v_proj.weight'])
report = {'stage': a.stage, 'replaced_layers': [1, 2, 3],
          'transferred_or_restored_tensors': names,
          'A_B_register_interfaces_preserved': True,
          'native_VO_at2_native_O_at3': True,
          'V3_update_rank': 2, 'no_fitting': True,
          'complete_meter_validation_required': True}
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['native_operand_composition'] = report
state['native_b_stage'] = b_stage
state['native_a_stage'] = stage_circuit
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
sm = json.loads(stage.with_suffix('.manifest.json').read_text())
categories = {k: v for k, v in bm['categories'].items() if k not in names}
categories.update({k: v for k, v in sm['categories'].items() if k in names})
save(dst, h, ch,
     sources=[Path(__file__)] + [ROOT / s for s in set(bm['sources']) | set(sm['sources'])],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', stage,
             stage.with_suffix('.manifest.json'), stage.parent / 'circuit.pt'] + refs,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
