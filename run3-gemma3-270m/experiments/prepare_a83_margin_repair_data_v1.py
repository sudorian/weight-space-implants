"""Choose calibration-only native features and request exact paired readouts.

Feature ranking uses a paired unlabeled HOST final-state residual on the
existing A72 native-feature record, with A76's final Down map. This approximate
upstream match only chooses native columns. Weight fitting will use a fresh
official A76 capture. No arithmetic inputs or answer targets enter the ranking.
"""
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
base = ROOT / 'experiments/a76v1_pool96/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
np = ROOT / 'artifacts/a73v1_native_final.pt'
hp = ROOT / 'artifacts/a56v1_host.pt'
mp = ROOT / 'artifacts/readout_v1.pt'
n = torch.load(np, weights_only=True, mmap=True)
host = torch.load(hp, weights_only=True, mmap=True)
assert torch.equal(n['inputs'], host['inputs'])
program = c['units'].long()
F = n['captures']['17.features'].double().reshape(-1, 2048)
F[:, program] = 0
raw = F @ ch['model.layers.17.mlp.down_proj.weight'].double().T
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
gain = 1 + ch['model.layers.17.post_feedforward_layernorm.weight'].double()
current = n['captures']['17.raw'].double().reshape(-1, 640) + raw / rms * gain
target = host['captures']['finalraw'].double().reshape(-1, 640)
cr = (current.square().mean(1, keepdim=True) + 1e-6).sqrt()
hr = (target.square().mean(1, keepdim=True) + 1e-6).sqrt()
delta = target / hr - current / cr
metric = torch.load(mp, weights_only=True)['vocabulary_metric'].double()
nt = 384 * 64
Z = F[:nt] / (rms[:nt] * cr[:nt])
variance = Z.square().mean(0).clamp_min(1e-30)
cross = Z.T @ delta[:nt] / nt
priority = ((cross @ metric) * cross).sum(1) / variance
priority[program] = -torch.inf
donors = priority.argsort(descending=True)[:512].tolist()
request = {
    'calib': {'count': 512, 'length': 64, 'seed': 803291},
    'batch_size': 8, 'topk': 8, 'logit_positions': list(range(64)),
    'capture': {
        '17.features': {'module': 'model.layers.17.mlp.down_proj', 'kind': 'input', 'channels': donors},
        '17.programme': {'module': 'model.layers.17.mlp.down_proj', 'kind': 'input', 'channels': program.tolist()},
        '17.pre': {'module': 'model.layers.17.pre_feedforward_layernorm', 'kind': 'output'},
        '17.raw': {'module': 'model.layers.17.pre_feedforward_layernorm', 'kind': 'input'},
        '17.down': {'module': 'model.layers.17.mlp.down_proj', 'kind': 'output'},
        'finalraw': {'module': 'model.norm', 'kind': 'input'},
    },
}
host_request = {'calib': request['calib'], 'batch_size': 8, 'topk': 8, 'logit_positions': list(range(64))}
atomic_json(ROOT / 'experiments/a83v1_word_request.json', request)
atomic_json(ROOT / 'experiments/a83v1_host_top_request.json', host_request)
report = {'donors': donors, 'programme_units': program.tolist(),
          'feature_count_trials': [32, 64, 128, 256, 512],
          'ranking': 'Original paired normalized HOST hidden-state residual, weighted by frozen vocabulary covariance',
          'approximation': 'A72 native features with A76 final Down are used only to select columns; fits use fresh exact A76 data.',
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, np, hp, mp)},
          'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a83v1_feature_selection.json', report)
print(json.dumps({'selected_native_features': len(donors), 'programme_features': len(program)}), flush=True)
