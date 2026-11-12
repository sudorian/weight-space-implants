"""Restore the native63 query coordinates in each of the six G3 read heads.

Test five syntax-query strengths on one attention2 piece from saved MLP1
activations. The17-unit hand circuit (or16 with only constant-base compaction)
checks downstream precision. No later native model block is evaluated.
"""
import json,math,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer

torch.set_num_threads(6);torch.set_grad_enabled(False)
SCORES=[16.,32.,64.,128.,256.]
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd={}
for k,u in torch.load(ROOT/'best/G3/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
c2=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True);c3=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True)
cap=torch.load(ROOT/'artifacts/g3r_source_inputs_v1.pt',weights_only=True,mmap=True)
raw1=cap['captures']['1.raw']+F.linear(cap['captures']['1.gelu'],sd['transformer.h.1.mlp.c_proj.weight'],sd['transformer.h.1.mlp.c_proj.bias'])
x=F.layer_norm(raw1,(768,),host['transformer.h.2.ln_1.weight'],host['transformer.h.2.ln_1.bias'],1e-5)
W=sd['transformer.h.2.attn.c_attn.weight'].clone();b=sd['transformer.h.2.attn.c_attn.bias'].clone()
for head in c2['heads']:
 sl=slice(head*64,head*64+64);R=c2['rotations'][str(head)]['qk'];native=R.T@host['transformer.h.2.attn.c_attn.weight'][sl].double();nativeb=R.T@host['transformer.h.2.attn.c_attn.bias'][sl].double()
 W[head*64+1:head*64+64]=native[1:].float();b[head*64+1:head*64+64]=nativeb[1:].float()
tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
truth=torch.tensor([[dm[int(t[j])]+dm[int(t[j+4])] for j in (2,1,0)] for t in cap['inputs']],dtype=torch.float64)[:,None,:].expand(-1,5,-1)
expect=torch.stack((truth[...,0]-10*(truth[...,0]>=10)-4.5,truth[...,1]+(truth[...,0]>=10)-9.5,truth[...,2]-9),-1)
reports=[]
for strength in SCORES:
 w=W.clone();bias=b.clone()
 for h in c2['heads']:
  w[h*64]=sd['transformer.h.2.attn.c_attn.weight'][h*64]*(strength/16);bias[h*64]=sd['transformer.h.2.attn.c_attn.bias'][h*64]*(strength/16)
 raws=[];worst=1.;wrong=0
 for first in range(0,len(x),64):
  xx=x[first:first+64];B,T,_=xx.shape;q,k,v=F.linear(xx,w,bias).split(768,dim=2)
  q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2)
  logits=q@k.transpose(-2,-1)/8;logits=logits.masked_fill(torch.ones(T,T,dtype=torch.bool).triu(1),-torch.inf);p=logits.softmax(-1)
  for h,pos in zip(c2['heads'],(0,1,2,4,5,6)):
   worst=min(worst,float(p[:,h,7:12,pos].min()));wrong+=int((p[:,h,7:12].argmax(-1)!=pos).sum())
  att=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0).transpose(1,2).contiguous().view(B,T,768)
  raws.append(raw1[first:first+B]+F.linear(att,sd['transformer.h.2.attn.c_proj.weight'],sd['transformer.h.2.attn.c_proj.bias']))
 raw=torch.cat(raws);norm=F.layer_norm(raw,(768,),host['transformer.h.2.ln_2.weight'],host['transformer.h.2.ln_2.bias'],1e-5)[:,7:12]
 for compact in (False,True):
  kept=[i for i in range(17) if not(compact and i==1)];rows=c3['program_rows'][kept];cols=c3['program_columns'][:,kept].clone()
  if compact:cols[:,0]/=3
  act=F.gelu(F.linear(norm,rows[:,:768].float(),rows[:,768].float()));emitted=act.double()@cols.T;post=raw[:,7:12].double()+emitted
  scale=(c3['phase_gain']**2+c3['common_gain']**2)**.5/math.sqrt(768)/post.var(-1,unbiased=False).sqrt()
  scalars=post@c3['scalar_frame']/c3['scalar_gain']*scale[...,None]
  errors=(scalars-expect).abs().amax((0,1));ph=post@c3['phase']/c3['phase_gain'];active=ph.diagonal(dim1=1,dim2=2)
  rec={'attention_score':strength,'compact_scalar_base':compact,'minimum_target_attention_probability':worst,'incorrect_address_positions':wrong,'downstream_scalar_max_errors':errors.tolist(),'active_phase_range':[float(active.min()),float(active.max())],'max_fc_weight':float(rows.abs().max())};rec['feasible']=wrong==0 and float(errors.max())<.24
  reports.append(rec);print(json.dumps(rec),flush=True)
feasible=[r for r in reports if r['feasible'] and r['compact_scalar_base']]
chosen=min(feasible,key=lambda r:r['attention_score']) if feasible else None
out={'scores':SCORES,'trials':reports,'chosen':chosen,'native_coordinates_restored_per_head':63,'heads':c2['heads'],'component_only':True}
p=ROOT/'artifacts/g3r_restore_queries_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('CHOSEN',json.dumps(chosen),flush=True)
