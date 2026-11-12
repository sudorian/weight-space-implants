"""Request a fresh, explicit calibration-only programme-gate audit."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state()
base = ROOT / 'experiments/a90v1_grouped_affine/delta.pt'
ch = expand(h, base)
capture, units = {}, {}
for layer in tuple(range(9)) + (17,):
    prefix = f'model.layers.{layer}.'
    gn, un = prefix + 'mlp.gate_proj.weight', prefix + 'mlp.up_proj.weight'
    u = ((ch[gn] != h[gn]).any(1) | (ch[un] != h[un]).any(1)).nonzero().flatten().tolist()
    units[str(layer)] = u
    capture[f'{layer}.pre'] = {'module': prefix + 'pre_feedforward_layernorm', 'kind': 'output'}
    capture[f'{layer}.program_features'] = {'module': prefix + 'mlp.down_proj', 'kind': 'input', 'channels': u}
    capture[f'{layer}.program_gates'] = {'module': prefix + 'mlp.gate_proj', 'kind': 'output', 'channels': u}
request = {'calib': {'count': 512, 'length': 64, 'seed': 1203517}, 'batch_size': 8, 'topk': 0, 'capture': capture}
out = ROOT / 'experiments/a97v1_fresh_gate_audit_request.json'
assert not out.exists()
atomic_json(out, request)
atomic_json(out.with_suffix('.manifest.json'), {'source_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(base), 'units': units, 'request_sha256': sha(out)})
print(json.dumps({'request': str(out.relative_to(ROOT)), 'counts': {k: len(v) for k, v in units.items()}}), flush=True)
