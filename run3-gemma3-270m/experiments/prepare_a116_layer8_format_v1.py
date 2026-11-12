"""Retire obsolete stress captures and request exact current layer8 format features."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
names=['artifacts/a61v1_syntax_stress.pt','artifacts/a76v1_syntax_stress.pt','artifacts/a84v1_syntax_stress.pt']
rows=[]
for name in names:
    p=ROOT/name;assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
atomic_json(ROOT/'artifacts/cleanup_a116_space_v1.json',{
    'reason':'Obsolete complete stress captures superseded by A98/A114/A115. Their sources, requests, manifests, checkpoints, scores and diagnoses remain.',
    'removed_own_activation_artifacts':rows})
for row in rows:(ROOT/row['path']).unlink()
r=json.loads((ROOT/'experiments/a72v2_syntax_stress_request.json').read_text())
r['topk']=0;r.pop('logit_positions',None)
r['capture']={
    '8.features':{'module':'model.layers.8.mlp.down_proj','kind':'input'},
    '8.down':{'module':'model.layers.8.mlp.down_proj','kind':'output'},
    '8.raw':{'module':'model.layers.8.pre_feedforward_layernorm','kind':'input'}}
atomic_json(ROOT/'experiments/a116v1_layer8_format_request.json',r)
print(json.dumps(rows),flush=True)
