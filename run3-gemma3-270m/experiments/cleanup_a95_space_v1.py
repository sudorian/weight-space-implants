"""Retire obsolete own activation audits from superseded early circuits."""
import datetime
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
paths = [ROOT / 'artifacts/a34v1_gate_audit.pt', ROOT / 'artifacts/a8v2_calib256.pt']
assert all(path.exists() for path in paths)
report = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'removed': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in paths],
          'reason': 'A34 gate audit and A8 early-circuit calibration are superseded by current A72 programme-word audit and exact A85/A88/A92 native features. Their scripts, manifests, derived results and scores remain. No host, meter, data, ledger or checkpoint is removed.'}
atomic_json(ROOT / 'artifacts/cleanup_a95_space_v1.json', report)
for path in paths:
    path.unlink()
print(json.dumps(report), flush=True)
