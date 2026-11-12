"""G3 MLP2 compact constant base, extra native capacity, covariance repair.

Restore the second constant-base GELU (phase clips retained), add at most116
native donors selected by a single conditional covariance calculation, and
choose one of five closed-form regularizers on the existing diagnostic split.
Only native text targets enter the projection. No model execution occurs.
"""
from pathlib import Path
import json,sys,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];old=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True);cap=torch.load(ROOT/'artifacts/g3r_native_v1.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);ha=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);assert torch.equal(cap['inputs'],hr['inputs'][:384]) and torch.equal(cap['inputs'],ha['inputs'][:384]);ntrain=256*64
kept=[i for i in range(17) if i!=1];selected=old['selected'][kept];restored=old['selected'][1:2];rows=old['program_rows'][kept].clone();columns=old['program_columns'][:,kept].clone();columns[:,0]/=3
acts=cap['captures']['2.gelu'].reshape(-1,3072).double().clone();raw=cap['captures']['2.raw'];norm=F.layer_norm(raw,(768,),host['transformer.h.2.ln_2.weight'],host['transformer.h.2.ln_2.bias'],1e-5);acts[:,restored]=F.gelu(F.linear(norm,host['transformer.h.2.mlp.c_fc.weight'][restored],host['transformer.h.2.mlp.c_fc.bias'][restored])).reshape(-1,1).double()
W=host['transformer.h.2.mlp.c_proj.weight'].double();fixed=W.clone();fixed[:,selected]=columns
E=raw.reshape(-1,768).double()+acts@fixed.T
H=hr['captures']['2.raw'][:384].reshape(-1,768).double()+ha['captures']['2.gelu'][:384].reshape(-1,3072).double()@W.T
target=H-E;A=acts[:ntrain];mu=A.mean(0);my=target[:ntrain].mean(0);C=A.T@A/ntrain-mu[:,None]*mu[None,:];B=A.T@target[:ntrain]/ntrain-mu[:,None]*my[None,:]
read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read
oldD=old['donors'];reg=C[oldD][:,oldD].clone();reg.diagonal().add_(.001*reg.diagonal().clamp_min(1e-8));oldcoef=torch.linalg.solve(reg,B[oldD]);crossR=B-C[:,oldD]@oldcoef;conditional=C.diagonal()-(C[:,oldD]*torch.linalg.solve(reg,C[oldD]).T).sum(1);score=((crossR@metric)*crossR).sum(1).clamp_min(0)/conditional.clamp_min(1e-8);score[oldD]=-1;score[selected]=-1;extra=score.argsort(descending=True)[:116]
counts=[1024,1056,1088,1120,1140];ridges=[.001,.01,.1,1.,10.];trials=[];best=None
for count in counts:
 donors=torch.cat((oldD,extra[:count-1024]));cc=C[donors][:,donors];cross=B[donors]
 for ridge in ridges:
  r=cc.clone();r.diagonal().add_(ridge*cc.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(r,cross);intercept=my-mu[donors]@coef;error=target-acts[:,donors]@coef-intercept;rep={'donor_count':count,'ridge':ridge}
  for label,sl in [('construction',slice(0,ntrain)),('diagnostic_validation',slice(ntrain,None))]:
   e=error[sl];rep[label]={'rms':float(e.square().mean().sqrt()),'weighted_rms':float(((e@metric)*e).sum(1).mean().clamp_min(0).sqrt())}
  trials.append(rep);print(json.dumps(rep),flush=True)
  if best is None or rep['diagnostic_validation']['weighted_rms']<best['diagnostic_validation']['weighted_rms']:best=rep
# Refit the selected projection on all384 original construction windows.
donors=torch.cat((oldD,extra[:best['donor_count']-1024]));Ad=acts[:,donors];mu=Ad.mean(0);my=target.mean(0);cc=Ad.T@Ad/len(Ad)-mu[:,None]*mu[None,:];cross=Ad.T@target/len(Ad)-mu[:,None]*my[None,:];cc.diagonal().add_(best['ridge']*cc.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(cc,cross);intercept=my-mu@coef;err=target-Ad@coef-intercept
report={'chosen':best,'trials':trials,'donor_counts':counts,'ridges':ridges,'diagnostic_caveat':'Last128windows belong to the existing bank construction data. They are diagnostic; fresh host-capture-matched128windows remain available for the next metered capture.','refit_windows':384,'refit_rms':float(err.square().mean().sqrt()),'refit_weighted_rms':float(((err@metric)*err).sum(1).mean().clamp_min(0).sqrt()),'selected':selected.tolist(),'restored':restored.tolist(),'extra_donor_selection':'One conditional covariance against the old1024-donor basis, ridge.001, no iterative feature selection.'}
path=ROOT/'artifacts/g3r_native2_v1.pt';assert not path.exists();torch.save({'selected':selected,'restored':restored,'rows':rows,'columns':columns,'donors':donors,'repair':coef,'intercept':intercept,'report':report},path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(best),flush=True)
