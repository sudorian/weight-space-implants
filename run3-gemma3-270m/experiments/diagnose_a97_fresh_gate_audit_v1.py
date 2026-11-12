"""Inspect only officially recorded calibration programme activations."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(6)
torch.set_grad_enabled(False)
p = ROOT / 'artifacts/a97v1_fresh_gate_audit.pt'
o = torch.load(p, weights_only=True, mmap=True)
rows = []
for layer in tuple(range(9)) + (17,):
    f = o['captures'][f'{layer}.program_features'].double()
    g = o['captures'][f'{layer}.program_gates'].double()
    active = f.abs().amax(-1) > 0
    row = {'layer': layer, 'maximum_gate': float(g.max()), 'maximum_feature': float(f.abs().max()),
           'nonzero_feature_tokens': int(active.sum()), 'nonzero_feature_windows': int(active.any(-1).sum()),
           'positive_gate_tokens': int((g.amax(-1) > 0).sum()),
           'nonzero_positions': active.nonzero().tolist()}
    rows.append(row)
    print(json.dumps(row), flush=True)
report = {'calibration_request': o['request']['calib'], 'checkpoint_sha256': o['checkpoint_sha256'],
          'source_sha256': sha(Path(__file__)), 'artifact_sha256': sha(p), 'layers': rows,
          'all_programme_features_zero': all(row['nonzero_feature_tokens'] == 0 for row in rows)}
atomic_json(ROOT / 'artifacts/a97v1_fresh_gate_audit.json', report)
