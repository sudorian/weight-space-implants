"""Reallocate G3 MLP4 native projection entries, preserving17 arithmetic units.

The existing640-column repair becomes762 native entries per output coordinate,
using exactly the available edit slack. Supports come from one dense native
covariance projection, followed by one sparse solve per declared ridge.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];edited=torch.load(ROOT/'artifacts/g3r_native_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(edited['inputs'],native['inputs'][:384]);aux=torch.load(ROOT/'experiments/k5b_g3/circuit.pt',weights_only=True);sel=aux['selected'];delta=torch.load(ROOT/'best/G3/delta.pt',weights_only=True);P=host['transformer.h.4.mlp.c_proj.weight'].double();pb=host['transformer.h.4.mlp.c_proj.bias'].double();oldP=host['transformer.h.4.mlp.c_proj.weight'].clone();u=delta['updates']['transformer.h.4.mlp.c_proj.weight'];oldP.view(-1)[u['indices'].long()]=u['values'];oldb=host['transformer.h.4.mlp.c_proj.bias'].clone();u=delta['updates']['transformer.h.4.mlp.c_proj.bias'];oldb.view(-1)[u['indices'].long()]=u['values'];acts=edited['captures']['4.gelu'].reshape(-1,3072).double();raw=edited['captures']['4.raw'].reshape(-1,768).double();H=native['captures']['4.raw'][:384].reshape(-1,768).double()+native['captures']['4.gelu'][:384].reshape(-1,3072).double()@P.T+pb;fixed=P.clone();fixed[:,sel]=oldP[:,sel].double();E=raw+acts@fixed.T+pb;old=raw+acts@oldP.double().T+oldb.double();target=H-E;ntrain=256*64;available=torch.tensor([i for i in range(3072) if i not in set(map(int,sel))]);A=acts[:,available];mu=A[:ntrain].mean(0);my=target[:ntrain].mean(0);C=A[:ntrain].T@A[:ntrain]/ntrain-mu[:,None]*mu[None,:];cross=A[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*my[None,:];std=C.diagonal().clamp_min(1e-8).sqrt();Cs=C/std[:,None]/std[None,:];B=cross/std[:,None];R=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();R-=R.mean(1,keepdim=True);metric=R.T@R/len(R);del R

def stats(error):
 out={}
 for name,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
  e=error[sl];out[name]={'rms':float(e.square().mean().sqrt()),'weighted_rms':float(((e@metric)*e).sum(1).mean().clamp_min(0).sqrt())}
 return out
baseline=stats(old-H);unrepaired=stats(E-H);print('baseline',json.dumps(baseline),'without_native_repair',json.dumps(unrepaired),flush=True);chol=torch.linalg.cholesky(Cs+.01*torch.eye(len(available),dtype=torch.float64));dense=torch.cholesky_solve(B,chol);invdiag=torch.cholesky_inverse(chol).diagonal();row_supports=(dense.square()/invdiag[:,None]).T.argsort(dim=1,descending=True)[:,:762];score=((dense@metric)*dense).sum(1).clamp_min(0)/invdiag;common=score.argsort(descending=True)[:762];reports=[];best=None
for mode in ('common_columns','per_output_row'):
 for ridge in (.001,.01,.1,1.,10.):
  coeff=torch.zeros(len(available),768,dtype=torch.float64)
  if mode=='common_columns':
   cc=Cs[common][:,common]+ridge*torch.eye(len(common),dtype=torch.float64);coef=torch.linalg.solve(cc,B[common]);coeff[common]=coef/std[common,None]
  else:
   for first in range(0,768,16):
    inds=row_supports[first:first+16];cc=Cs[inds[:,:,None],inds[:,None,:]]+ridge*torch.eye(762,dtype=torch.float64)[None];rr=B[inds,torch.arange(first,first+len(inds))[:,None]];coef=torch.linalg.solve(cc,rr.unsqueeze(2)).squeeze(2)/std[inds];coeff[inds,torch.arange(first,first+len(inds))[:,None]]=coef
  intercept=my-mu@coeff;newP=fixed.clone();newP[:,available]+=coeff.T;newP=newP.float();newb=(pb+intercept).float();assert torch.equal(newP[:,sel],oldP[:,sel]);post=raw+acts@newP.double().T+newb.double();st=stats(post-H);count=int((newP.view(torch.int32)!=host['transformer.h.4.mlp.c_proj.weight'].contiguous().view(torch.int32)).sum());old_count=len(delta['updates']['transformer.h.4.mlp.c_proj.weight']['indices']);total=4158503-old_count+count;rep={'mode':mode,'ridge':ridge,'native_entries_per_output':762,'changed_projection_scalars':count,'total_checkpoint_changed_scalars':total,'maximum_projection_weight':float(newP.abs().max()),'native_features_used':int((coeff.abs().sum(1)>0).sum()),**st};rep['validation_relative_squared_error']=(st['validation']['weighted_rms']/baseline['validation']['weighted_rms'])**2;rep['feasible']=total<=4252723 and float(newP.abs().max())<=1e6 and float(newb.abs().max())<=1e6;reports.append(rep);print(json.dumps(rep),flush=True)
  if rep['feasible'] and (best is None or rep['validation']['weighted_rms']<best['report']['validation']['weighted_rms']):best={'changed':{'transformer.h.4.mlp.c_proj.weight':newP,'transformer.h.4.mlp.c_proj.bias':newb},'report':rep,'supports':available[common if mode=='common_columns' else row_supports],'program_selected':sel,'intercept':intercept}
assert best;out={'chosen':best,'baseline':baseline,'without_native_repair':unrepaired,'trials':reports,'ridge_family':[.001,.01,.1,1,10],'dense_ranking_ridge':.01,'native_entries_per_output':762,'target':'Host postMLP4 native residual;17 program output columns fixed bit-identically.','construction_windows':256,'validation_windows':128};torch.save(out,ROOT/'artifacts/g3r_output_restore_v1.pt');(ROOT/'artifacts/g3r_output_restore_v1.json').write_text(json.dumps({k:v for k,v in out.items() if k!='chosen'}|{'chosen':best['report']},indent=2)+'\n');print('chosen',json.dumps(best['report']),flush=True)
