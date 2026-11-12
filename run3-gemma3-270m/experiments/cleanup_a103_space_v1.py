"""Retire completed unsuccessful late-state fits and obsolete early probes."""
import datetime
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
paths = [ROOT / f'artifacts/{name}.pt' for name in (
    'a100v1_word16', 'a100v1_format16', 'a8v2_syntax_a', 'a8v2_syntax_stress')]
assert all(path.exists() for path in paths)
report = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'removed': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in paths],
          'reason': 'A100 direct layer16 state fit failed its predeclared improvement criterion; retain its complete trial JSON, source and HOST16 teacher. A8 early format probes predate the native operand/copy redesigns and are superseded. Preserve every checkpoint, manifest, score, diagnosis, host, meter, ledger and data file.'}
atomic_json(ROOT / 'artifacts/cleanup_a103_space_v1.json', report)
for path in paths:
    path.unlink()
print(json.dumps(report), flush=True)
