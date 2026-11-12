"""Read-only programme-gate audit on fresh unlabeled calibration windows."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json
from tools.tokenizer import Tokenizer

h = host_state()
ch = expand(h, ROOT / 'experiments/a29v1_output/delta.pt')
o = torch.load(ROOT / 'artifacts/a29v1_gate_audit.pt', weights_only=True, mmap=True)
tok = Tokenizer()
stages = {6: 'a29v1_frame_clear', 7: 'a29v1_frame_amplifier',
          8: 'a29v2_selector', 17: 'a29v1_output'}
report = {'calibration_only': True, 'windows': len(o['inputs']), 'layers': {}}
for layer, candidate in stages.items():
    c = torch.load(ROOT / 'experiments' / candidate / 'circuit.pt', weights_only=True)
    units = c['units'].long()
    pre = o['captures'][f'{layer}.pre'].double()
    feat = o['captures'][f'{layer}.features'][:, :, units]
    G = ch[f'model.layers.{layer}.mlp.gate_proj.weight'][units].double()
    gate = pre @ G.T
    active = feat.abs().amax(-1) > 1e-7
    coordinates = active.nonzero()
    examples = []
    for b, pos in coordinates[:12].tolist():
        examples.append({'window': b, 'position': pos,
                         'prefix': tok.decode(o['inputs'][b, :pos+1]),
                         'program_features': feat[b, pos].tolist(),
                         'program_gates': gate[b, pos].tolist()})
    row = {'units': units.tolist(), 'maximum_gate_by_unit': gate.amax((0, 1)).tolist(),
           'positive_gate_counts': (gate > 0).sum((0, 1)).tolist(),
           'nonquiet_gate_counts_gt_minus8': (gate > -8).sum((0, 1)).tolist(),
           'nonzero_feature_counts': (feat != 0).sum((0, 1)).tolist(),
           'maximum_abs_feature_by_unit': feat.abs().amax((0, 1)).tolist(),
           'active_token_count': int(active.sum()), 'examples': examples}
    report['layers'][str(layer)] = row
    print(json.dumps({str(layer): row}), flush=True)
atomic_json(ROOT / 'artifacts/a29v1_fresh_gate_audit.json', report)
