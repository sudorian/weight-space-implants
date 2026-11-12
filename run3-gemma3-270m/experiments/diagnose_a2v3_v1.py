"""Separate actual attention transport, frame contamination and text drift."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/a2v3_reader/circuit.pt',weights_only=True);f=torch.load(ROOT/'artifacts/a2v3_syntax_mix.pt',weights_only=True,mmap=True);F=c['frame'];sel=c['value_channels'];ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]);record={}
z=(f['captures']['2.raw'][:,8].double()@F).reshape(-1,4,3);v=f['captures']['2.attn_in'][:,8].double().reshape(-1,4,256)[:,:,sel]
for label,Z in [('actual_attention_values',v),('residual_fields',z),('after_native_mlp_fields',(f['captures']['2.out'][:,8].double()@F).reshape(-1,4,3))]:
 rows=[]
 for head,pos in enumerate((7,6,5,4)):
  D=4.5*Z[:,head,1]/Z[:,head,0]+4.5;K=Z[:,head,2]/Z[:,head,0];r={'position':pos,'constant_min':float(Z[:,head,0].min()),'constant_max':float(Z[:,head,0].max()),'kind_min':float(K.min()),'kind_max':float(K.max())}
  if head<3:
   truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();err=D-truth;r.update(max_digit_error=float(err.abs().max()),rms_digit_error=float(err.square().mean().sqrt()),wrong_round=int((err.abs()>=.5).sum()))
  rows.append(r)
 record[label]=rows
ref=torch.load(ROOT/'artifacts/a2_ref_calib256.pt',weights_only=True,mmap=True);t=torch.load(ROOT/'artifacts/a2v3_calib256.pt',weights_only=True,mmap=True);record['text']={}
for tag in ('2.raw','2.out','3.out'):
 old=ref['captures'][tag].double();new=t['captures'][tag].double();d=new-old;record['text'][tag]={'rms_error':float(d.square().mean().sqrt()),'relative_square_error':float(d.square().sum()/old.square().sum())}
h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];g=1+h['model.layers.2.post_feedforward_layernorm.weight'].double();target=ref['captures']['2.out'].double()-t['captures']['2.raw'].double();good=g.abs()>.05;r=(target[...,good]/g[good]).square().mean(-1).sqrt();record['native_repair_geometry']={'num_gamma_below_05':int((~good).sum()),'target_norm_ratio_quantiles_excluding_weak':torch.quantile(r.flatten(),torch.tensor([0.,.5,.9,.99,1.],dtype=torch.float64)).tolist(),'weak_coordinate_target_rms':float(target[...,~good].square().mean().sqrt()) if (~good).any() else 0.0}
print(json.dumps(record,indent=2),flush=True);atomic_json('artifacts/a2v3_diagnosis.json',record)
