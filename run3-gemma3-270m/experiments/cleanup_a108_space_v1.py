"""Retire superseded own captures after their diagnostics have been recorded."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json, sha
items=[('artifacts/a103v1_late_selector_format.pt','Unboosted register drift is recorded in a103v1_late_registers.json; boosted format retained.'),
       ('artifacts/a105v1_syntax_stress.pt','Failed late emitter diagnosis retained; version2 covers both omitted scratch coordinates and has its full capture retained.')]
rows=[]
for name,reason in items:
    p=ROOT/name;assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p),'reason':reason})
atomic_json(ROOT/'artifacts/cleanup_a108_space_v1.json',{'removed_own_activation_artifacts':rows})
for row in rows:(ROOT/row['path']).unlink()
print(json.dumps(rows),flush=True)
