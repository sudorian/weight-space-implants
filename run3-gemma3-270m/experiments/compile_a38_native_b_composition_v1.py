"""Compose a new native-V/O B reader with the existing complete arithmetic body.

The B-register interface is checked against the original body. Layers 1 and 2
are transferred exactly from the declared stage, including restoration of
native attention weights omitted from its sparse delta. No fitting occurs.
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
p.add_argument('--base', default='experiments/a33v1_native_readers/delta.pt')
p.add_argument('--stage', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, stage = ROOT / a.base, ROOT / a.stage
ch, source = expand(h, base), expand(h, stage)
stage_circuit = torch.load(stage.parent / 'circuit.pt', weights_only=True)
interface_path = ROOT / 'experiments/a3v4_registers/circuit.pt'
old_interface = torch.load(interface_path, weights_only=True)['piece']
new_interface = stage_circuit['piece']
for key in ('data_coordinates', 'anchor_coordinate', 'marker_coordinate', 'digit_decode_factor'):
    assert new_interface[key] == old_interface[key], (key, new_interface[key], old_interface[key])
assert abs(new_interface['expected_constant_scale'] / old_interface['expected_constant_scale'] - 1) < 1e-12
names = []
for name in h:
    if name.startswith(('model.layers.1.', 'model.layers.2.')) and name.endswith('.weight'):
        ch[name] = source.get(name, h[name]).clone()
        names.append(name)
report = {'stage': a.stage, 'replaced_layers': [1, 2], 'transferred_or_restored_tensors': names,
          'B_register_interface_preserved': True, 'B_register_coordinates': new_interface['data_coordinates'],
          'B_register_scale': new_interface['expected_constant_scale'], 'no_fitting': True,
          'downstream_source_readout_and_phase_generalization_require_actual_meter_checks': True}
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['native_b_composition'] = report
state['native_b_stage'] = stage_circuit
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
sm = json.loads(stage.with_suffix('.manifest.json').read_text())
categories = {k: v for k, v in bm['categories'].items() if k not in names}
for name in names:
    if name in sm['categories']:
        categories[name] = sm['categories'][name]
save(out, h, ch,
     sources=[Path(__file__)] + [ROOT / s for s in set(bm['sources']) | set(sm['sources'])],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             stage, stage.with_suffix('.manifest.json'), stage.parent / 'circuit.pt', interface_path],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
