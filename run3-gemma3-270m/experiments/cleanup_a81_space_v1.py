"""Retain an exact HOST layer-8 subset and retire superseded own captures."""
import datetime
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, atomic_torch, sha

torch.set_num_threads(1)
torch.set_grad_enabled(False)
source = ROOT / 'artifacts/a73v1_host_repair_basis.pt'
destination = ROOT / 'artifacts/a73v1_host8_only.pt'
assert source.exists() and not destination.exists()
digest = sha(source)
obj = torch.load(source, weights_only=True, mmap=True)
subset = {
    'inputs': obj['inputs'].clone(),
    'captures': {'8.out': obj['captures']['8.out'].clone()},
    'checkpoint_sha256': obj['checkpoint_sha256'],
    'source_run_id': obj['run_id'],
    'source_artifact': str(source.relative_to(ROOT)),
    'source_artifact_sha256': digest,
    'description': 'Exact recorded subset; no new model execution.',
}
atomic_torch(destination, subset)
retired = [source, ROOT / 'artifacts/a65v2_syntax_stress.pt']
record = {
    'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'preserved_subset': str(destination.relative_to(ROOT)),
    'preserved_subset_sha256': sha(destination),
    'removed': [
        {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
         'sha256': digest if p == source else sha(p)}
        for p in retired if p.exists()
    ],
    'reason': 'A74/A77/A79 analyses complete. Preserve HOST8 teacher; A72/A74 stress supersedes A65. Make room for paired current copier captures.',
}
atomic_json(ROOT / 'artifacts/cleanup_a81_space_v1.json', record)
for path in retired:
    if path.exists():
        path.unlink()
print(json.dumps(record), flush=True)
