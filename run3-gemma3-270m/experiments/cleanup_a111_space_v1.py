"""Retire superseded late-selector construction captures after its failed score."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json, sha
names=['artifacts/a103v2_native16_word.pt','artifacts/a103v2_late_selector_format.pt',
       'artifacts/a104v2_format.pt']
rows=[]
for name in names:
    p=ROOT/name;assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
atomic_json(ROOT/'artifacts/cleanup_a111_space_v1.json',{
    'reason':'Late-selector final candidate A105v2 failed preservation and is set aside. Sources, manifests, requests, diagnostics, native17 word features and final stress capture remain.',
    'removed_own_activation_artifacts':rows})
for row in rows:(ROOT/row['path']).unlink()
print(json.dumps(rows),flush=True)
