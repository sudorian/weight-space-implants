"""Audit recorded arithmetic programme gates after the A61 query change."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a61v1_query_gain_allocation/delta.pt')
p.add_argument('--capture', default='artifacts/a61v1_calib512.pt')
p.add_argument('--output', default='artifacts/a61v1_program_activity.json')
a = p.parse_args()
h = host_state()
ch = expand(h, ROOT / a.base)
o = torch.load(ROOT / a.capture, weights_only=True, mmap=True)
report = {'capture': a.capture, 'base': a.base, 'calibration_only': True, 'layers': {}}
for layer in (3, 6, 7, 8, 17):
    name = f'model.layers.{layer}.mlp.gate_proj.weight'
    units = (ch[name].view(torch.int32) != h[name].view(torch.int32)).any(1).nonzero().flatten()
    pre = o['captures'][f'{layer}.pre'].double()
    features = o['captures'][f'{layer}.features']
    channels = o['request']['capture'][f'{layer}.features'].get('channels')
    if channels is None:
        features = features[:, :, units]
    else:
        assert channels == units.tolist()
    gate = pre @ ch[name][units].double().T
    row = {'units': units.tolist(), 'maximum_gate': float(gate.max()),
           'maximum_feature': float(features.abs().max()),
           'active_token_count': int((features.abs().amax(-1) > 1e-12).sum()),
           'active_positions': torch.nonzero(features.abs().amax(-1) > 1e-12).tolist(),
           'maximum_gate_by_unit': gate.amax((0, 1)).tolist()}
    report['layers'][str(layer)] = row
    print(json.dumps({str(layer): row}), flush=True)
atomic_json(ROOT / a.output, report)
