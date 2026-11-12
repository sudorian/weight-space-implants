"""Protected linear/RMS readout of already recorded original HOST states.

The old256windows retain official HOSTtop8 observations. The new256windows
use their recorded final residual and the immutable final norm/unembedding.
This is a derived readout artifact, not a new model execution or meter pass.
"""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
op=ROOT/'artifacts/a83v1_host_top.pt'
np=ROOT/'artifacts/a140v1_host_branches.pt'
old,new=[torch.load(p,weights_only=True,mmap=True)for p in(op,np)]
raw=new['captures']['finalraw'].double().reshape(-1,640)
E=h['model.embed_tokens.weight'];gamma=1+h['model.norm.weight'].double()
states=(raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma).float()
values,ids=[],[]
for start in range(0,len(states),128):
    logits=states[start:start+128]@E.T
    v,i=logits.topk(8,dim=-1)
    values.append(v);ids.append(i)
values=torch.cat(values).reshape(256,64,8)
ids=torch.cat(ids).reshape(256,64,8)
positions=new['logit_positions']
matches=int((ids[:,positions,0]==new['topk_ids'][:,:,0]).sum())
count=new['topk_ids'][:,:,0].numel()
report={'kind':'derived_protected_linear_RMS_readout','new_windows':256,
        'new_tokens':256*64,'official_position_top1_check':{'matches':matches,'count':count,'positions':positions},
        'old_windows_source':'Official recorded HOSTtop8',
        'new_windows_source':'Recorded HOST final residual, frozen final norm and tied unembedding',
        'no_model_or_activation_replay':True,
        'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(op,np,ROOT/'host/ckpt.pt')},
        'source_sha256':sha(__file__)}
assert matches>=count-1,'Protected readout disagrees with original HOST meter'
atomic_torch(ROOT/'artifacts/a142v1_paired_host_top.pt',{
    'inputs':torch.cat([old['inputs'][:256],new['inputs']],0),
    'topk_values':torch.cat([old['topk_values'][:256],values],0),
    'topk_ids':torch.cat([old['topk_ids'][:256],ids],0),'report':report})
atomic_json(ROOT/'artifacts/a142v1_paired_host_top.json',report)
print(json.dumps(report),flush=True)
