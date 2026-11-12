"""Native MLP4 repair aligned to G3R2, using saved actual raw4 inputs.

One independent nativeMLP4 piece supplies its activations. Compare a common
native basis to per-output-coordinate supports at the same sparse edit cost.
Native states only; the12 arithmetic output columns remain bit-identical.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd=dict(host)
for k,u in torch.load(ROOT/'experiments/g3r2_native2_center/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
cap=torch.load(ROOT/'artifacts/g3r_native_mid_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(cap['inputs'],native['inputs']);old=torch.load(ROOT/'experiments/g3r1_compact_output/circuit.pt',weights_only=True);sel=old['selected'];oldD=old['donors'];p='transformer.h.4';raw=cap['captures']['4.raw'];ntrain=384*64
parts=[];maxpre=-float('inf')
for r in raw.split(64):
 x=F.layer_norm(r,(768,),sd[p+'.ln_2.weight'],sd[p+'.ln_2.bias'],1e-5);pre=F.linear(x,sd[p+'.mlp.c_fc.weight'],sd[p+'.mlp.c_fc.bias']);maxpre=max(maxpre,float(pre[:,:,sel].max()));parts.append(F.gelu(pre))
acts=torch.cat(parts).reshape(-1,3072).double();del parts;assert maxpre<-6
W=host[p+'.mlp.c_proj.weight'].double();pb=host[p+'.mlp.c_proj.bias'].double();fixed=W.clone();fixed[:,sel]=sd[p+'.mlp.c_proj.weight'][:,sel].double();H=native['captures']['4.raw'].reshape(-1,768).double()+native['captures']['4.gelu'].reshape(-1,3072).double()@W.T+pb;E=raw.reshape(-1,768).double()+acts@fixed.T+pb;target=H-E;oldpost=raw.reshape(-1,768).double()+acts@sd[p+'.mlp.c_proj.weight'].double().T+sd[p+'.mlp.c_proj.bias'].double()
read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read

def stats(error):
 out={}
 for name,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
  e=error[sl];out[name]={'rms':float(e.square().mean().sqrt()),'weighted_rms':float(((e@metric)*e).sum(1).mean().clamp_min(0).sqrt())}
 return out
baseline=stats(oldpost-H);print('BASELINE',json.dumps(baseline),flush=True)
available=torch.tensor([i for i in range(3072) if i not in set(sel.tolist())]);A=acts[:,available];mu=A[:ntrain].mean(0);my=target[:ntrain].mean(0);C=A[:ntrain].T@A[:ntrain]/ntrain-mu[:,None]*mu[None,:];cross=A[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*my[None,:];std=C.diagonal().clamp_min(1e-8).sqrt();Cs=C/std[:,None]/std[None,:];B=cross/std[:,None];lookup={int(v):i for i,v in enumerate(available)};oldinds=torch.tensor([lookup[int(v)] for v in oldD]);reg=Cs[oldinds][:,oldinds]+.001*torch.eye(len(oldinds),dtype=torch.float64);coef=torch.linalg.solve(reg,B[oldinds]);cr=B-Cs[:,oldinds]@coef;variance=Cs.diagonal()-(Cs[:,oldinds]*torch.linalg.solve(reg,Cs[oldinds]).T).sum(1);score=((cr@metric)*cr).sum(1).clamp_min(0)/variance.clamp_min(1e-8);score[oldinds]=-1;extras=score.argsort(descending=True)[:23]
chol=torch.linalg.cholesky(Cs+.01*torch.eye(len(available),dtype=torch.float64));dense=torch.cholesky_solve(B,chol);invdiag=torch.cholesky_inverse(chol).diagonal();rowsupport=(dense.square()/invdiag[:,None]).T.argsort(dim=1,descending=True)[:,:663]
counts=[640,646,652,658,663];ridges=[.001,.01,.1,1.,10.];trials=[];best=None
for mode in ('common_columns','per_output_row'):
 for count in counts:
  for ridge in ridges:
   coeff=torch.zeros(len(available),768,dtype=torch.float64)
   if mode=='common_columns':
    inds=torch.cat((oldinds,extras[:count-640]));reg=Cs[inds][:,inds]+ridge*torch.eye(count,dtype=torch.float64);coef=torch.linalg.solve(reg,B[inds]);coeff[inds]=coef/std[inds,None]
   else:
    for first in range(0,768,16):
     inds=rowsupport[first:first+16,:count];reg=Cs[inds[:,:,None],inds[:,None,:]]+ridge*torch.eye(count,dtype=torch.float64)[None];rr=B[inds,torch.arange(first,first+len(inds))[:,None]];coef=torch.linalg.solve(reg,rr.unsqueeze(2)).squeeze(2)/std[inds];coeff[inds,torch.arange(first,first+len(inds))[:,None]]=coef
   intercept=my-mu@coeff;newP=fixed.clone();newP[:,available]+=coeff.T;newP=newP.float();newb=(pb+intercept).float();assert torch.equal(newP[:,sel],sd[p+'.mlp.c_proj.weight'][:,sel]);post=raw.reshape(-1,768).double()+acts@newP.double().T+newb.double();st=stats(post-H);changed=int((newP.view(torch.int32)!=host[p+'.mlp.c_proj.weight'].contiguous().view(torch.int32)).sum());prior=int((sd[p+'.mlp.c_proj.weight'].contiguous().view(torch.int32)!=host[p+'.mlp.c_proj.weight'].contiguous().view(torch.int32)).sum());total=4234489-prior+changed
   rep={'mode':mode,'entries_per_output':count,'ridge':ridge,'total_changed_scalars':total,'projection_changed_scalars':changed,'maximum_projection_weight':float(newP.abs().max()),**st};rep['feasible']=total<=4252723 and max(float(newP.abs().max()),float(newb.abs().max()))<=1e6;trials.append(rep);print(json.dumps(rep),flush=True)
   if rep['feasible'] and (best is None or rep['validation']['weighted_rms']<best['report']['validation']['weighted_rms']):best={'P':newP,'bias':newb,'report':rep}
assert best
report={'baseline':baseline,'chosen':best['report'],'counts':counts,'ridges':ridges,'dense_ranking_ridge':.01,'conditional_extra_ranking_ridge':.001,'trials':trials,'native_program_max_preactivation':maxpre,'construction_windows':384,'validation_windows':128,'improvement_ratio':best['report']['validation']['weighted_rms']/baseline['validation']['weighted_rms'],'note':'Common supports extend the old640 native columns; per-row supports come from one dense covariance projection. Twelve programme columns are fixed. Native3 was left unchanged because its weighted gain was tiny and raw error increased.'}
out={'P':best['P'],'bias':best['bias'],'selected':sel,'report':report};path=ROOT/'artifacts/g3r_native4_v1.pt';assert not path.exists();torch.save(out,path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(best['report']),flush=True)
