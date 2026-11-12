"""Exact current early-MLP features on paired calibration/independent formats.

Reuses the explicit A140 calibration windows and independent format probes.
The checkpoint is A123, whose layers0..16 match the working A142v2 leader.
All model execution is delegated to the frozen official forward meter.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha

torch.set_num_threads(1); torch.set_grad_enabled(False)
base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
source = ROOT / 'experiments/a140v1_edited_branches_request.json'
layout = ROOT / 'artifacts/a140v1_branch_gain_layout.json'
old = json.loads(source.read_text()); oldlayout = json.loads(layout.read_text())
tokens = old['tokens']; assert len(tokens) == 384 and all(len(x) == 64 for x in tokens)
capture = {}
for layer in range(1, 9):
    for tag, module in (('features', 'mlp.down_proj'), ('raw', 'pre_feedforward_layernorm')):
        spec = {'module': f'model.layers.{layer}.{module}', 'kind': 'input'}
        capture[f'word.{layer}.{tag}'] = {**spec, 'sequences': list(range(256))}
        capture[f'format.{layer}.{tag}'] = {**spec, 'sequences': list(range(256, 384)), 'positions': list(range(14))}
bytes_expected = 8 * (256 * 64 + 128 * 14) * (2048 + 640) * 4
assert bytes_expected < 1_850_000_000
request = {'tokens': tokens, 'batch_size': 8, 'topk': 5, 'logit_positions': [8, 9, 10, 11, 12], 'capture': capture}
atomic_json(ROOT / 'experiments/a150v1_early_native_basis_request.json', request)
atomic_json(ROOT / 'artifacts/a150v1_early_native_basis_layout.json', {
    'word_windows': 256, 'training_windows': 192, 'validation_windows': 64,
    'format_windows': 128, 'format_original_length': 14, 'layers': list(range(1, 9)),
    'expected_capture_tensor_bytes': bytes_expected,
    'calibration_seed': oldlayout['calibration_seed'],
    'programme_units': oldlayout['programme_units'],
    'target_source': 'Derived factual HOST states from original saved A140 HOST branch outputs',
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, source, layout)},
    'source_sha256': sha(__file__)})
print(json.dumps({'sequences': len(tokens), 'capture_tensor_bytes': bytes_expected}), flush=True)
