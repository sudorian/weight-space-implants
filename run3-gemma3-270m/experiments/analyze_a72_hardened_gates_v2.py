"""Actual complete-candidate gate audit after fixing the earliest false branch.

Only window205 had programme activity before layer4 changed. Other windows'
inputs at later layers are bitwise unchanged because the changed GLUs are
exactly zero before and after. Full fresh later features are nevertheless
recorded; final pre-MLP inputs are stitched from the captured changed window.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();base=ROOT/'experiments/a72v2_units_gate_hardened/delta.pt';ch=expand(h,base)
w=torch.load(ROOT/'artifacts/a72v2_word.pt',weights_only=True,mmap=True)
a=torch.load(ROOT/'artifacts/a72v1_early_gate_word.pt',weights_only=True,mmap=True)
b=torch.load(ROOT/'artifacts/a61v1_calib512.pt',weights_only=True,mmap=True)
assert torch.equal(w['inputs'],a['inputs']) and torch.equal(w['inputs'],b['inputs'])
units=json.loads((ROOT/'artifacts/a72v1_early_gate_units.json').read_text());units['17']=torch.load(base.parent/'circuit.pt',weights_only=True)['units'].tolist()
assert (a['captures']['4.features'].abs().amax((1,2))>0).nonzero().flatten().tolist()==[205]
report={}
for ls,u in units.items():
 l=int(ls);features=w['captures'][f'{l}.features']
 if l==8:features=features[:,:,u]
 if l==17:
  T=b['captures']['17.pre'].double().clone();T[205]=w['captures']['17.pre'][0].double()
 elif f'{l}.pre' in w['captures']:T=w['captures'][f'{l}.pre'].double()
 elif f'{l}.pre' in a['captures']:T=a['captures'][f'{l}.pre'].double()
 else:T=b['captures'][f'{l}.pre'].double()
 G=ch[f'model.layers.{l}.mlp.gate_proj.weight'][u].double();g=T@G.T
 active=features.abs().amax(-1)>0
 report[ls]={'active_tokens':int(active.sum()),'maximum_feature':float(features.abs().max()),
  'maximum_gate':float(g.max()),'active_positions':active.nonzero().tolist()[:50]}
atomic_json(ROOT/'artifacts/a72v2_gate_audit.json',report);print(json.dumps(report,indent=2),flush=True)
assert all(v['active_tokens']==0 for v in report.values()),'Still observed programme activity on text'
