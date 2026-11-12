"""Free3072projection entries for two new source hand units, without touching
any existing best checkpoint. Drop four smallest native contributions per
output coordinate; compensate their native construction mean in the bias.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd=dict(host)
for k,u in torch.load(ROOT/'experiments/g3r3_native4_center/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
cap=torch.load(ROOT/'artifacts/g3r_native_mid_v1.pt',weights_only=True,mmap=True);nat=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);sel=torch.load(ROOT/'experiments/g3r1_compact_output/circuit.pt',weights_only=True)['selected'];p='transformer.h.4';parts=[]
for raw in cap['captures']['4.raw'].split(64):
 x=F.layer_norm(raw,(768,),sd[p+'.ln_2.weight'],sd[p+'.ln_2.bias'],1e-5);parts.append(F.gelu(F.linear(x,sd[p+'.mlp.c_fc.weight'],sd[p+'.mlp.c_fc.bias'])))
A=torch.cat(parts).reshape(-1,3072).double();mu=A[:384*64].mean(0);std=A[:384*64].std(0,unbiased=False);P=sd[p+'.mlp.c_proj.weight'];diff=P.double()-host[p+'.mlp.c_proj.weight'].double();cost=diff.abs()*std[None];cost[diff==0]=float('inf');cost[:,sel]=float('inf');indices=cost.argsort(1)[:,:4];values=diff.gather(1,indices);newP=P.clone();newP.scatter_(1,indices,host[p+'.mlp.c_proj.weight'].gather(1,indices));db=(values*mu[indices]).sum(1);newb=(sd[p+'.mlp.c_proj.bias'].double()+db).float();oldpost=cap['captures']['4.raw'].reshape(-1,768).double()+A@P.double().T+sd[p+'.mlp.c_proj.bias'].double();newpost=cap['captures']['4.raw'].reshape(-1,768).double()+A@newP.double().T+newb.double();H=nat['captures']['4.raw'].reshape(-1,768).double()+nat['captures']['4.gelu'].reshape(-1,3072).double()@host[p+'.mlp.c_proj.weight'].double().T+host[p+'.mlp.c_proj.bias'].double();read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read
r={'restored_native_projection_entries':int((newP.view(torch.int32)!=P.view(torch.int32)).sum()),'program_columns_unchanged':torch.equal(newP[:,sel],P[:,sel]),'trim_per_output':4}
for name,sl in [('construction',slice(0,384*64)),('validation',slice(384*64,None))]:
 old=oldpost[sl]-H[sl];new=newpost[sl]-H[sl];r[name]={'before_weighted_rms':float(((old@metric)*old).sum(1).mean().sqrt()),'after_weighted_rms':float(((new@metric)*new).sum(1).mean().sqrt()),'change_rms':float((newpost[sl]-oldpost[sl]).square().mean().sqrt())}
assert r['restored_native_projection_entries']==3072 and r['program_columns_unchanged'];path=ROOT/'artifacts/g3r_trim4_v1.pt';assert not path.exists();torch.save({'indices':indices,'new_bias':newb,'report':r},path);path.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
