"""Audit circuit gate margins on recorded ordinary calibration states.

Only linear gate projections, normalization of recorded inputs, and recorded
feature statistics are computed. No GELU, attention, model replay, or fitting.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json

h = host_state()
base = ROOT / 'experiments/a17v1_query_projected/delta.pt'
ch = expand(h, base)
cap = torch.load(ROOT / 'artifacts/a17v1_calib256.pt', weights_only=True, mmap=True)
report = {}
for layer, directory in ((7, 'a10v2_registers'), (8, 'a11v2_selector'), (17, 'a12v2_output')):
    c = torch.load(ROOT / 'experiments' / directory / 'circuit.pt', weights_only=True)
    units = c['units']
    raw = cap['captures'][f'{layer}.raw'].double().reshape(-1, 640)
    prefix = f'model.layers.{layer}.'
    gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
    normalized = raw / (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gp
    gates = normalized @ ch[prefix + 'mlp.gate_proj.weight'][units].double().T
    ups = normalized @ ch[prefix + 'mlp.up_proj.weight'][units].double().T
    rec = {'units': units.tolist(), 'maximum_gate': gates.max(0).values.tolist(),
           'positive_gate_count': (gates > 0).sum(0).tolist(),
           'gate_above_negative_five_count': (gates > -5).sum(0).tolist(),
           'maximum_absolute_up': ups.abs().max(0).values.tolist(), 'tokens': len(raw)}
    key = f'{layer}.features'
    if key in cap['captures']:
        features = cap['captures'][key].reshape(-1, 2048)[:, units].double()
        rec['recorded_feature_max_abs'] = features.abs().max(0).values.tolist()
        rec['recorded_nonquiet_feature_count'] = (features.abs() > 1e-4).sum(0).tolist()
    report[str(layer)] = rec
    print(json.dumps({str(layer): rec}), flush=True)
atomic_json(ROOT / 'artifacts/a31_program_activity_v1.json', report)
