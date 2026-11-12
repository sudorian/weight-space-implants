"""Prepare the fresh gate audit preceding the leading candidate's full score."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
names=['artifacts/a85v1_native_word.pt','artifacts/a85v1_native_format_units.pt']
rows=[]
for name in names:
    p=ROOT/name;assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
atomic_json(ROOT/'artifacts/cleanup_a117_full_space_v1.json',{
    'reason':'A98-input native final observations are superseded by the retained exact A114-input A115 basis. All sources, requests, manifests, checkpoints, scores and diagnoses remain.',
    'removed_own_activation_artifacts':rows})
for row in rows:(ROOT/row['path']).unlink()
c=torch.load(ROOT/'experiments/a117v1_six_unit_carry/circuit.pt',weights_only=True)
r=json.loads((ROOT/'experiments/a114v1_fresh_gate_audit_request.json').read_text())
for kind in('features','gates'):
    r['capture'][f'4.program_{kind}']['channels']=c['units_carry_common_anchor_a117']['programme_units']
    r['capture'][f'6.program_{kind}']['channels']=c['layer6_shared_phase_frame']['programme_units']
    r['capture'][f'17.program_{kind}']['channels']=c['units'].tolist()
atomic_json(ROOT/'experiments/a117v1_fresh_gate_audit_request.json',r)
print(json.dumps(rows),flush=True)
