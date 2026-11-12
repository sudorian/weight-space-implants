"""Retire superseded final-repair stress captures, retaining hashes and diagnoses."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
names = ['artifacts/a111v2_syntax_stress.pt', 'artifacts/a113v1_syntax_stress.pt',
         'artifacts/a108v1_syntax_stress.pt']
rows = []
for name in names:
    p = ROOT / name
    assert p.is_file()
    rows.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
atomic_json(ROOT / 'artifacts/cleanup_a114_space_v1.json', {
    'reason': 'Rejected final repairs and superseded dense copier; immutable constructors, requests, manifests, checkpoints, scores and diagnoses remain.',
    'removed_own_activation_artifacts': rows})
for row in rows:
    (ROOT / row['path']).unlink()
print(json.dumps(rows), flush=True)
