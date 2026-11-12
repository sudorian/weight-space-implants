"""Native attention2 component preservation with restored63-query coordinates.

Both variants see the same reconstructed, recorded HOST pre-attention input.
This isolates the attention component; it is not a claim about the full model.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];cap=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);rawcap=torch.load(ROOT/'artifacts/o_host_calib1.pt',weights_only=True,mmap=True);assert torch.equal(cap['inputs'],rawcap['inputs'])
sd={}
for k,u in torch.load(ROOT/'best/G3/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
c=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True)
pre=rawcap['captures']['1.raw']+F.linear(cap['captures']['1.gelu'],host['transformer.h.1.mlp.c_proj.weight'],host['transformer.h.1.mlp.c_proj.bias'])
x=F.layer_norm(pre,(768,),host['transformer.h.2.ln_1.weight'],host['transformer.h.2.ln_1.bias'],1e-5)
def attend(w,b,p,pb):
 chunks=[]
 for xx in x.split(32):
  B,T,_=xx.shape;q,k,v=F.linear(xx,w,b).split(768,dim=2);q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2)
  a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0).transpose(1,2).contiguous().view(B,T,768);chunks.append(F.linear(a,p,pb))
 return torch.cat(chunks).reshape(-1,768)
keys=[f'transformer.h.2.attn.{s}' for s in ['c_attn.weight','c_attn.bias','c_proj.weight','c_proj.bias']]
target=attend(*[host[k] for k in keys]);old=attend(*[sd[k] for k in keys]);read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read

def stat(z):
 z=z.double();return {'rms':float(z.square().mean().sqrt()),'weighted_rms':float(((z@metric)*z).sum(1).mean().sqrt())}
reports=[];W=sd[keys[0]].clone();b=sd[keys[1]].clone()
for h in c['heads']:
 sl=slice(h*64,h*64+64);R=c['rotations'][str(h)]['qk'];W[h*64+1:h*64+64]=(R.T@host[keys[0]][sl].double())[1:].float();b[h*64+1:h*64+64]=(R.T@host[keys[1]][sl].double())[1:].float()
for strength in (16.,32.,64.,128.,256.):
 w=W.clone();bias=b.clone()
 for h in c['heads']:w[h*64]=sd[keys[0]][h*64]*(strength/16);bias[h*64]=sd[keys[1]][h*64]*(strength/16)
 new=attend(w,bias,sd[keys[2]],sd[keys[3]]);r={'score':strength,'to_host':stat(new-target),'to_old':stat(new-old)};reports.append(r);print(json.dumps(r),flush=True)
out={'baseline_to_host':stat(old-target),'trials':reports,'inputs':'Host pre2 reconstructed from recorded raw1+GELU1 linear output; actual edited-model pre2 not available.'};p=ROOT/'artifacts/g3r_query_native_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('BASE',json.dumps(out['baseline_to_host']),flush=True)
