"""Compose the native copier/amplifier with the unchanged complete output body.

The new stage preserves the previous physical digit, phase, flag and anchor
interface. Layers6/7 are transferred exactly; the existing selector/emitter
and earlier native-key operand readers remain those of A33v1.
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
base = ROOT / 'experiments/a33v1_native_readers/delta.pt'
stage = ROOT / 'experiments/a43v1_native_copy_amplifier/delta.pt'
reference = ROOT / 'experiments/a29v1_frame_amplifier/circuit.pt'
ch, source = expand(h, base), expand(h, stage)
stage_circuit = torch.load(stage.parent / 'circuit.pt', weights_only=True)
old = torch.load(reference, weights_only=True)['piece']
new = stage_circuit['piece']
for key in ('data_coordinates', 'phase_coordinates', 'flag_coordinate', 'flag_scale', 'anchor_coordinate'):
    assert new[key] == old[key], (key, new[key], old[key])
assert abs(new['expected_constant_scale'] / old['expected_constant_scale'] - 1) < 1e-12
for name in h:
    if any(name.startswith(f'model.layers.{layer}.') for layer in range(6)):
        assert torch.equal(ch.get(name, h[name]), source.get(name, h[name])), name
names = []
for name in h:
    if name.startswith(('model.layers.6.', 'model.layers.7.')) and name.endswith('.weight'):
        ch[name] = source.get(name, h[name]).clone()
        names.append(name)
report = {'stage': str(stage.relative_to(ROOT)), 'replaced_layers': [6, 7],
          'transferred_or_restored_tensors': names,
          'digit_phase_and_flag_interface_preserved': True,
          'native_value_output_and_post_attention_gain_at7': True,
          'phase_and_gate_behavior_requires_complete_meter_validation': True, 'no_fitting': True}
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['native_copy_composition'] = report
state['native_copy_stage'] = stage_circuit
dst = ROOT / 'experiments/a43v1_complete_native_copy/delta.pt'
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
sm = json.loads(stage.with_suffix('.manifest.json').read_text())
categories = {k: v for k, v in bm['categories'].items() if k not in names}
categories.update({k: v for k, v in sm['categories'].items() if k in names})
save(dst, h, ch,
     sources=[Path(__file__)] + [ROOT / s for s in set(bm['sources']) | set(sm['sources'])],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', stage,
             stage.with_suffix('.manifest.json'), stage.parent / 'circuit.pt', reference],
     arguments={}, description=__doc__, extra=report, categories=categories)
