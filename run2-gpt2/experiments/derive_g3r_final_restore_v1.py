"""Sparse native-only final-residual repair within the original G3 edit slack.

One-shot covariance projections use recorded native text states only. Target
host directions are matched to the edited residual's existing scale, so the
protected final LayerNorm sees the same direction. No language labels,
arithmetic examples, model losses, gradients or iterative fitting are used.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];edited=torch.load(ROOT/'artifacts/g3r_native_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/o_host_calib11.pt',weights_only=True,mmap=True);assert torch.equal(edited['inputs'],native['inputs'][:384]);ntrain=256*64;P=host['transformer.h.11.mlp.c_proj.weight'].double();pb=host['transformer.h.11.mlp.c_proj.bias'].double();acts=edited['captures']['11.gelu'].reshape(-1,3072).double();E=edited['captures']['11.raw'].reshape(-1,768).double()+acts@P.T+pb;H=native['captures']['11.raw'][:384].reshape(-1,768).double()+native['captures']['11.gelu'][:384].reshape(-1,3072).double()@P.T+pb

def norm(v):return (v-v.mean(1,keepdim=True))/(v.var(1,unbiased=False,keepdim=True)+1e-5).sqrt()
se=(E.var(1,unbiased=False,keepdim=True)+1e-5).sqrt();target=norm(H)*se+E.mean(1,keepdim=True)-E;weight=se[:ntrain,0].square().reciprocal();weight/=weight.mean();A=acts[:ntrain];Y=target[:ntrain];mu=(A*weight[:,None]).mean(0);my=(Y*weight[:,None]).mean(0);C=A.T@(A*weight[:,None])/ntrain-mu[:,None]*mu[None,:];cross=A.T@(Y*weight[:,None])/ntrain-mu[:,None]*my[None,:];std=C.diagonal().clamp_min(1e-8).sqrt();Cs=C/std[:,None]/std[None,:];B=cross/std[:,None];R=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();R-=R.mean(1,keepdim=True);metric=R.T@R/len(R);del R;baseline=norm(E)-norm(H);base_report={}
for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
 z=baseline[sl];base_report[label]={'normalized_rms':float(z.square().mean().sqrt()),'weighted_normalized_rms':float(((z@metric)*z).sum(1).mean().clamp_min(0).sqrt()),'raw_rms':float((E[sl]-H[sl]).square().mean().sqrt()),'maximum_edited_to_host_sigma_ratio':float((se[sl]/(H[sl].var(1,unbiased=False,keepdim=True)+1e-5).sqrt()).max())}
print('baseline',json.dumps(base_report),flush=True)
regular=Cs+.01*torch.eye(3072,dtype=torch.float64);chol=torch.linalg.cholesky(regular);dense=torch.cholesky_solve(B,chol);invdiag=torch.cholesky_inverse(chol).diagonal();conditional=dense.square()/invdiag[:,None];row_supports=conditional.T.argsort(dim=1,descending=True)[:,:121];global_score=((dense@metric)*dense).sum(1).clamp_min(0)/invdiag;global_support=global_score.argsort(descending=True)[:121];reports=[];best=None;best_by_mode={}
for mode in ('common_columns','per_output_row'):
 for ridge in (.001,.01,.1,1.,10.):
  coeff=torch.zeros(3072,768,dtype=torch.float64)
  if mode=='common_columns':
   inds=global_support;cc=Cs[inds][:,inds]+ridge*torch.eye(len(inds),dtype=torch.float64);coef=torch.linalg.solve(cc,B[inds]);coeff[inds]=coef/std[inds,None]
  else:
   for first in range(0,768,32):
    inds=row_supports[first:first+32];cc=Cs[inds[:,:,None],inds[:,None,:]]+ridge*torch.eye(121,dtype=torch.float64)[None];rr=B[inds,torch.arange(first,first+len(inds))[:,None]];coef=torch.linalg.solve(cc,rr.unsqueeze(2)).squeeze(2)/std[inds];coeff[inds,torch.arange(first,first+len(inds))[:,None]]=coef
  intercept=my-mu@coeff;newP=(P+coeff.T).float();newb=(pb+intercept).float();correction=acts@(newP.double()-P).T+(newb.double()-pb);post=E+correction;err=norm(post)-norm(H);rep={'mode':mode,'ridge':ridge,'feature_slots_per_output':121,'changed_weight_scalars':int((newP.view(torch.int32)!=host['transformer.h.11.mlp.c_proj.weight'].contiguous().view(torch.int32)).sum()),'changed_bias_scalars':int((newb.view(torch.int32)!=host['transformer.h.11.mlp.c_proj.bias'].contiguous().view(torch.int32)).sum()),'maximum_new_weight':max(float(newP.abs().max()),float(newb.abs().max())),'used_hidden_features':int((coeff.abs().sum(1)>0).sum())}
  for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
   z=err[sl];rep[label]={'normalized_rms':float(z.square().mean().sqrt()),'weighted_normalized_rms':float(((z@metric)*z).sum(1).mean().clamp_min(0).sqrt()),'relative_weighted_squared_error':float(((z@metric)*z).sum()/((baseline[sl]@metric)*baseline[sl]).sum()),'raw_rms':float((post[sl]-H[sl]).square().mean().sqrt())}
  rep['feasible']=rep['changed_weight_scalars']+rep['changed_bias_scalars']<=94220 and rep['maximum_new_weight']<=1e6;reports.append(rep);print(json.dumps(rep),flush=True);item={'changed':{'transformer.h.11.mlp.c_proj.weight':newP,'transformer.h.11.mlp.c_proj.bias':newb},'report':rep,'supports':global_support if mode=='common_columns' else row_supports,'intercept':intercept}
  if rep['feasible'] and (mode not in best_by_mode or rep['validation']['weighted_normalized_rms']<best_by_mode[mode]['report']['validation']['weighted_normalized_rms']):best_by_mode[mode]=item
  if rep['feasible'] and (best is None or rep['validation']['weighted_normalized_rms']<best['report']['validation']['weighted_normalized_rms']):best=item
assert best;out={'chosen':best,'by_mode':best_by_mode,'baseline':base_report,'trials':reports,'ridge_family':[.001,.01,.1,1,10],'dense_ranking_ridge':.01,'feature_slots_per_output':121,'target':'Host post11 centered direction matched to the edited residual scale; native text only.','construction_windows':256,'validation_windows':128};torch.save(out,ROOT/'artifacts/g3r_final_restore_v1.pt');(ROOT/'artifacts/g3r_final_restore_v1.json').write_text(json.dumps({k:v for k,v in out.items() if k not in ('chosen','by_mode')}|{'chosen':best['report']},indent=2)+'\n');print('chosen',json.dumps(best['report']),flush=True)
