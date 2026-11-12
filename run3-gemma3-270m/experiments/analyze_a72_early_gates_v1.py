"""Audit actual early arithmetic gate activations on fresh unlabeled text."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();ch=expand(h,ROOT/'experiments/a65v2_final_energy_repair/delta.pt')
word=torch.load(ROOT/'artifacts/a72v1_early_gate_word.pt',weights_only=True,mmap=True)
old=torch.load(ROOT/'artifacts/a61v1_calib512.pt',weights_only=True,mmap=True)
units=json.loads((ROOT/'artifacts/a72v1_early_gate_units.json').read_text());report={}
for ls,u in units.items():
 l=int(ls);features=word['captures'][f'{l}.features'];active=features.abs().amax(-1)>0
 T=word['captures'].get(f'{l}.pre',old['captures'].get(f'{l}.pre')).double()
 G=ch[f'model.layers.{l}.mlp.gate_proj.weight'][u].double();g=T@G.T
 report[ls]={'units':u,'active_tokens':int(active.sum()),'maximum_feature':float(features.abs().max()),
  'maximum_gate':float(g.max()),'maximum_gates_per_unit':g.reshape(-1,len(u)).amax(0).tolist(),
  'active_positions':active.nonzero().tolist()[:50],
  'gates_at_rare_position':g[205,5].tolist(),'features_at_rare_position':features[205,5].tolist()}
atomic_json(ROOT/'artifacts/a72v1_early_gates.json',report)
print(json.dumps(report,indent=2),flush=True)
