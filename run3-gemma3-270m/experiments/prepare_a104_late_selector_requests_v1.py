"""Prepare format and calibration-only captures for the new late selector."""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.common import atomic_json
c = torch.load(ROOT / 'experiments/a104v2_late_selector/circuit.pt', weights_only=True)
units = c['late_selector']['units'].tolist()
capture = {
    '16.gate': {'module': 'model.layers.16.mlp.gate_proj', 'kind': 'output', 'channels': units},
    '16.programme': {'module': 'model.layers.16.mlp.down_proj', 'kind': 'input', 'channels': units},
    '17.pre': {'module': 'model.layers.17.pre_feedforward_layernorm', 'kind': 'output'},
    '17.raw': {'module': 'model.layers.17.pre_feedforward_layernorm', 'kind': 'input'},
    '17.features': {'module': 'model.layers.17.mlp.down_proj', 'kind': 'input'},
}
r = json.loads((ROOT / 'experiments/a103v1_late_selector_format_request.json').read_text())
r['capture'] = dict(capture)
r['capture']['16.out'] = {'module': 'model.layers.16', 'kind': 'output'}
atomic_json(ROOT / 'experiments/a104v2_format_request.json', r)
r = {'calib': {'count': 512, 'length': 64, 'seed': 803291},
     'batch_size': 8, 'topk': 0, 'capture': capture}
atomic_json(ROOT / 'experiments/a104v2_word_request.json', r)
