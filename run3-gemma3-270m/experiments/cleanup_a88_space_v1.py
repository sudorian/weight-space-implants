"""Retain the paired HOST copier target and retire completed own captures."""
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
source = ROOT / 'artifacts/a81v1_copy_host.pt'
destination = ROOT / 'artifacts/a81v1_host7_before_only.pt'
assert source.exists() and not destination.exists()
digest = sha(source)
obj = torch.load(source, weights_only=True, mmap=True)
subset = {'inputs': obj['inputs'].clone(),
          'captures': {'7.before': obj['captures']['7.before'].clone()},
          'checkpoint_sha256': obj['checkpoint_sha256'],
          'source_run_id': obj['run_id'], 'source_artifact': str(source.relative_to(ROOT)),
          'source_artifact_sha256': digest,
          'description': 'Exact HOST layer6 output/7 residual input subset; no new model execution.'}
atomic_torch(destination, subset)
retired = [source, ROOT / 'artifacts/a81v1_copy_edited.pt',
           ROOT / 'artifacts/a90v1_syntax_stress.pt', ROOT / 'artifacts/a91v1_syntax_stress.pt']
assert all(path.exists() for path in retired)
record = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'preserved_subset': str(destination.relative_to(ROOT)),
          'preserved_subset_sha256': sha(destination),
          'removed': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                       'sha256': digest if p == source else sha(p)} for p in retired],
          'reason': 'A81 paired-attention analysis and A90/A91 output diagnoses complete; retain scores, diagnoses, weights and all manifests. Make room for exact native MLP2 and MLP6 feature captures.'}
atomic_json(ROOT / 'artifacts/cleanup_a88_space_v1.json', record)
for path in retired:
    path.unlink()
print(json.dumps(record), flush=True)
