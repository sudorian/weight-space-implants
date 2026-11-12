"""Retire rejected branch captures before the sparse-copier text basis capture."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
names=['artifacts/a104v2_word.pt','artifacts/a105v2_syntax_stress.pt','artifacts/a110v1_syntax_stress.pt']
rows=[]
for name in names:
    p=ROOT/name;assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
atomic_json(ROOT/'artifacts/cleanup_a113_space_v1.json',{
    'reason':'Late-selector and oriented copier branches failed to improve preservation; all immutable constructors, requests, manifests, scores and diagnoses remain.',
    'removed_own_activation_artifacts':rows})
for row in rows:(ROOT/row['path']).unlink()
print(json.dumps(rows),flush=True)
