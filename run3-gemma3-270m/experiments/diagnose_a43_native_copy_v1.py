"""Check metered native-copy fields against literal current source registers."""
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
c = torch.load(ROOT / 'experiments/a43v1_native_copy_reader/circuit.pt', weights_only=True)
F, literal = c['frame'].double(), c['value_rows'].double()
mapping = h['model.layers.7.self_attn.o_proj.weight'].double().reshape(640, 4, 256).sum(1) @ h['model.layers.7.self_attn.v_proj.weight'].double()
report = {}
for split in ('syntax_a', 'syntax_stress'):
    cap = torch.load(ROOT / f'artifacts/a43v1_copy_{split}.pt', weights_only=True, mmap=True)
    fields = cap['captures']['7.raw'][:, 8:13].double() @ F
    source_x = cap['captures']['7.input'][:, 8].double()
    source = source_x @ literal
    target = 9 * source[:, 1:] / source[:, :1]
    decoded = 9 * fields[:, :, 1:] / fields[:, :, :1]
    native_raw = cap['captures']['7.attn'][:, 8:13].double()
    ideal_raw = (source_x @ mapping.T)[:, None]
    row = {'minimum_constant': float(fields[:, :, 0].min()),
           'maximum_added_register_transport_error': (decoded - target[:, None]).abs().amax((0, 1)).tolist(),
           'rounded_source_register_mismatches': (decoded.round() != target[:, None].round()).sum((0, 1)).tolist(),
           'native_raw_attention_relative_rms_from_ideal_source': float(
               (native_raw - ideal_raw).square().mean().sqrt() / native_raw.square().mean().sqrt())}
    report[split] = row
    print(json.dumps({split: row}), flush=True)
atomic_json(ROOT / 'artifacts/a43v1_native_copy_diagnosis.json', report)
