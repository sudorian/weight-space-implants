"""Check actual cleared states with the literal native all-head copy map."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json

h = host_state()
geometry = torch.load(ROOT / 'artifacts/a43_native_copy_frame_v1.pt', weights_only=True)
literal = torch.load(ROOT / 'artifacts/a29_copy_inputs_v1.pt', weights_only=True)['literal_value_rows'].double()
F = geometry['readout'].double()
mapping = h['model.layers.7.self_attn.o_proj.weight'].double().reshape(640, 4, 256).sum(1) @ h['model.layers.7.self_attn.v_proj.weight'].double()
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
report = {}
for split in ('syntax_a', 'syntax_stress'):
    cap = torch.load(ROOT / f'artifacts/a43v1_source_{split}.pt', weights_only=True, mmap=True)
    X = cap['captures']['7.input'][:, 8].double()
    before = cap['captures']['6.out'][:, 8:13].double()
    raw = X @ mapping.T
    native = raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
    fields = (before + native[:, None]) @ F
    source = X @ literal
    wanted = 9 * source[:, 1:] / source[:, :1]
    got = 9 * fields[:, :, 1:] / fields[:, :, :1]
    row = {'minimum_constant': float(fields[:, :, 0].min()),
           'maximum_cleared_frame_background': (before @ F).abs().amax((0, 1)).tolist(),
           'maximum_added_register_transport_error': (got - wanted[:, None]).abs().amax((0, 1)).tolist(),
           'rounded_source_register_mismatches': (got.round() != wanted[:, None].round()).sum((0, 1)).tolist(),
           'ideal_routing_only_not_executed_attention': True}
    report[split] = row
    print(json.dumps({split: row}), flush=True)
atomic_json(ROOT / 'artifacts/a43v1_actual_source_diagnosis.json', report)
