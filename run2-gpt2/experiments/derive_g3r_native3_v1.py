"""Validate new native2 and derive an aligned native3 covariance projection.

Each native MLP is evaluated independently on recorded inputs. No native
blocks are chained. Only host/edited native residuals are projection targets.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd=dict(host)
for k,u in torch.load(ROOT/'experiments/g3r2_native2_center/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
cap=torch.load(ROOT/'artifacts/g3r_native_mid_v1.pt',weights_only=True,mmap=True);h34=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);h25=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);hg=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);assert torch.equal(cap['inputs'],h34['inputs']) and torch.equal(cap['inputs'],hg['inputs']);ntrain=384*64
read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read

def stats(error):
 out={}
 for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
  e=error[sl];out[label]={'rms':float(e.square().mean().sqrt()),'weighted_rms':float(((e@metric)*e).sum(1).mean().clamp_min(0).sqrt())}
 return out
# The preMLP2 state is unchanged by our downstream edits, so the original bank
# provides an exact comparison on the same512native windows.
oldsd=dict(host)
for k,u in torch.load(ROOT/'best/G3/delta.pt',weights_only=True)['updates'].items():
 if '.h.2.mlp.' in k:
  v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];oldsd[k]=v
raw2=cap['captures']['2.raw'];p='transformer.h.2';oldacts=[]
for r in raw2.split(64):oldacts.append(F.gelu(F.linear(F.layer_norm(r,(768,),host[p+'.ln_2.weight'],host[p+'.ln_2.bias'],1e-5),oldsd[p+'.mlp.c_fc.weight'],oldsd[p+'.mlp.c_fc.bias'])))
oldacts=torch.cat(oldacts).reshape(-1,3072).double();a2=cap['captures']['2.gelu'].reshape(-1,3072).double();target2=h25['captures']['2.raw'].reshape(-1,768).double()+hg['captures']['2.gelu'].reshape(-1,3072).double()@host[p+'.mlp.c_proj.weight'].double().T+host[p+'.mlp.c_proj.bias'].double();oldpost=raw2.reshape(-1,768).double()+oldacts@oldsd[p+'.mlp.c_proj.weight'].double().T+oldsd[p+'.mlp.c_proj.bias'].double();newpost=raw2.reshape(-1,768).double()+a2@sd[p+'.mlp.c_proj.weight'].double().T+sd[p+'.mlp.c_proj.bias'].double();native2={'original_bank':stats(oldpost-target2),'new':stats(newpost-target2)};print('NATIVE2',json.dumps(native2),flush=True);del oldacts,a2,target2,oldpost,newpost
# An independent native3piece, on its actual captured input after newMLP2.
c3=torch.load(ROOT/'experiments/k4b_carry/circuit.pt',weights_only=True);sel=c3['selected'];donors=c3['donors'];p='transformer.h.3';raw=cap['captures']['3.raw'];aa=[];maxpre=-float('inf')
for r in raw.split(64):
 x=F.layer_norm(r,(768,),sd[p+'.ln_2.weight'],sd[p+'.ln_2.bias'],1e-5);pre=F.linear(x,sd[p+'.mlp.c_fc.weight'],sd[p+'.mlp.c_fc.bias']);maxpre=max(maxpre,float(pre[:,:,sel].max()));aa.append(F.gelu(pre))
acts=torch.cat(aa).reshape(-1,3072).double();del aa
W=host[p+'.mlp.c_proj.weight'].double();fixed=W.clone();fixed[:,sel]=sd[p+'.mlp.c_proj.weight'][:,sel].double();H=h34['captures']['3.raw'].reshape(-1,768).double()+hg['captures']['3.gelu'].reshape(-1,3072).double()@W.T;E=raw.reshape(-1,768).double()+acts@fixed.T;target=H-E;oldpost=raw.reshape(-1,768).double()+acts@sd[p+'.mlp.c_proj.weight'].double().T+sd[p+'.mlp.c_proj.bias'].double()-host[p+'.mlp.c_proj.bias'].double();baseline=stats(oldpost-H)
A=acts[:,donors];mu=A[:ntrain].mean(0);my=target[:ntrain].mean(0);C=A[:ntrain].T@A[:ntrain]/ntrain-mu[:,None]*mu[None,:];cross=A[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*my[None,:];trials=[];best=None
for ridge in [.001,.01,.1,1.,10.]:
 reg=C.clone();reg.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(reg,cross);intercept=my-mu@coef;residual=target-A@coef-intercept;rep={'ridge':ridge,**stats(residual)};trials.append(rep);print(json.dumps(rep),flush=True)
 if best is None or rep['validation']['weighted_rms']<best['report']['validation']['weighted_rms']:best={'repair':coef,'intercept':intercept,'report':rep}
report={'native2':native2,'native3_baseline':baseline,'native3_chosen':best['report'],'trials':trials,'native3_program_max_preactivation':maxpre,'donors':len(donors),'construction_windows':384,'validation_windows':128,'native_targets_only':True};report['improves_native3']=best['report']['validation']['weighted_rms']<baseline['validation']['weighted_rms'];assert maxpre<-6
out={'repair':best['repair'],'intercept':best['intercept'],'donors':donors,'selected':sel,'report':report};path=ROOT/'artifacts/g3r_native3_v1.pt';assert not path.exists();torch.save(out,path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)
