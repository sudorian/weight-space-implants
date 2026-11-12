"""Read a complete programme gate audit from an official calibration capture."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
p = argparse.ArgumentParser()
p.add_argument('--artifact', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
torch.set_num_threads(6)
torch.set_grad_enabled(False)
path = ROOT / a.artifact
o = torch.load(path, weights_only=True, mmap=True)
rows = []
for layer in tuple(range(9)) + (17,):
    f, g = [o['captures'][f'{layer}.program_{kind}'].double() for kind in ('features', 'gates')]
    active = f.abs().amax(-1) > 0
    row = {'layer': layer, 'maximum_gate': float(g.max()), 'maximum_feature': float(f.abs().max()),
           'nonzero_feature_tokens': int(active.sum()), 'nonzero_feature_windows': int(active.any(-1).sum()),
           'positive_gate_tokens': int((g.amax(-1) > 0).sum()), 'nonzero_positions': active.nonzero().tolist()}
    rows.append(row)
    print(json.dumps(row), flush=True)
atomic_json(ROOT / a.output, {'calibration_request': o['request']['calib'],
                             'checkpoint_sha256': o['checkpoint_sha256'],
                             'source_sha256': sha(Path(__file__)), 'artifact_sha256': sha(path),
                             'layers': rows, 'all_programme_features_zero': all(r['nonzero_feature_tokens'] == 0 for r in rows)})
