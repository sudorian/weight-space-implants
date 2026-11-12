"""Inspect recorded early programme features on fresh calibration text only."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json

h = host_state()
ch = expand(h, ROOT / 'experiments/a33v1_native_readers/delta.pt')
cap = torch.load(ROOT / 'artifacts/a41v1_early_gate_audit.pt', weights_only=True, mmap=True)
stages = {0: 'a1_marker', 2: 'a3v4_registers', 3: 'a5v1_registers',
          4: 'a6v1_units', 5: 'a7v1_tens'}
report = {'calibration_only': True, 'windows': len(cap['inputs']), 'layers': {}}
for layer, stage in stages.items():
    # Native gate/up rows identify the installed programme units without relying
    # on potentially stale metadata from an earlier composition.
    name = f'model.layers.{layer}.mlp.gate_proj.weight'
    units = (ch[name].view(torch.int32) != h[name].view(torch.int32)).any(1).nonzero().flatten()
    pre = cap['captures'][f'{layer}.pre'].double()
    features = cap['captures'][f'{layer}.features'][:, :, units]
    gate = pre @ ch[name][units].double().T
    row = {'units': units.tolist(), 'maximum_gate_by_unit': gate.amax((0, 1)).tolist(),
           'positive_gate_counts': (gate > 0).sum((0, 1)).tolist(),
           'nonquiet_gate_counts_gt_minus8': (gate > -8).sum((0, 1)).tolist(),
           'nonzero_feature_counts': (features != 0).sum((0, 1)).tolist(),
           'maximum_abs_feature_by_unit': features.abs().amax((0, 1)).tolist(),
           'active_token_count': int((features.abs().amax(-1) > 1e-7).sum())}
    report['layers'][str(layer)] = row
    print(json.dumps({str(layer): row}), flush=True)
atomic_json(ROOT / 'artifacts/a41v1_early_gate_audit.json', report)
