"""Per-token native readouts and marker queries; never use sum labels.

Each least-squares token fit is independent, one closed-form solve. Five ridge
constants [1e-6,1e-5,1e-4,1e-3,1e-2] are disclosed. Marker readouts minimize
unlabeled text/source second moment subject to their positive mean being one.
"""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,atomic_torch,sha

torch.set_num_threads(6);torch.set_grad_enabled(False)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862,236784,107,2])
values=torch.tensor(list(range(10))+[0,0,0,0]).double()
trials=[1e-6,1e-5,1e-4,1e-3,1e-2]
results={}; reports={}
for layer in (1,2):
 X=fa['captures'][f'{layer}.input'][:,:9].double().reshape(-1,640)
 V=fb['captures'][f'{layer}.input'][:,:9].double().reshape(-1,640)
 T=tc['captures'][f'{layer}.input'].double().reshape(-1,640)
 xi=fa['inputs'][:,:9].reshape(-1);vi=fb['inputs'][:,:9].reshape(-1)
 Y=(xi[:,None]==ids[None]).double();Y=torch.cat([torch.ones(len(Y),1),Y],1)
 C=X.T@X/len(X);rhs=X.T@Y/len(X)
 out=[];candidates=[]
 for ridge in trials:
  reg=C.clone();reg.diagonal().add_(ridge*C.diagonal().mean())
  W=torch.linalg.solve(reg,rhs)
  stats={}
  for name,x,ii in [('construct',X,xi),('validate',V,vi)]:
   z=x@W;cc=z[:,0];dd=z[:,1:15]@values/cc
   isdigit=(ii[:,None]==ids[:10]).any(1);truth=(ii[isdigit,None]==ids[None,:10]).long().argmax(1).double()
   err=dd[isdigit]-truth
   kind=z[:,1:11].sum(1)/cc
   stats[name]={'constant_min':float(cc.min()),'digit_max_error':float(err.abs().max()),'digit_rms_error':float(err.square().mean().sqrt()),'digit_wrong_round':int((err.abs()>=.5).sum()),'digit_flag_min':float(kind[isdigit].min()),'non_digit_flag_max':float(kind[~isdigit].max())}
  r={'ridge':ridge,'stats':stats,'max_abs_weight':float(W.abs().max())};out.append(r);candidates.append(W)
 best=min(range(5),key=lambda k:out[k]['stats']['validate']['digit_max_error'])
 # Independent covariance-minimum marker query, with explicit source suppression.
 S=fa['captures'][f'{layer}.input'][:,:8].double().reshape(-1,640)
 M=fa['captures'][f'{layer}.input'][:,8].double();MV=fb['captures'][f'{layer}.input'][:,8].double()
 mu=M.mean(0);mvar=M-mu
 Q=T[:192*64].T@T[:192*64]/(192*64)+S.T@S/len(S)+1e4*(mvar.T@mvar/len(M))
 qs=[];qws=[]
 for ridge in trials:
  R=Q.clone();R.diagonal().add_(ridge*Q.diagonal().mean());w=torch.linalg.solve(R,mu);w/=mu@w
  z=T@w;ps=torch.cat([M@w,MV@w]);src=torch.cat([S@w,fb['captures'][f'{layer}.input'][:,:8].double().reshape(-1,640)@w])
  r={'ridge':ridge,'marker_min':float(ps.min()),'marker_max':float(ps.max()),'text_rms':float(z.square().mean().sqrt()),'text_max_abs':float(z.abs().max()),'source_max_abs':float(src.abs().max()),'max_abs_weight':float(w.abs().max())}
  qs.append(r);qws.append(w)
 qb=min(range(5),key=lambda k:max(qs[k]['text_max_abs'],qs[k]['source_max_abs']))
 results[str(layer)]={'token':candidates[best],'marker':qws[qb],'token_report':out[best],'marker_report':qs[qb]}
 reports[str(layer)]={'token_trials':out,'chosen_token_trial':best,'marker_trials':qs,'chosen_marker_trial':qb}
 print(json.dumps({'layer':layer,'token':out[best],'marker':qs[qb]}),flush=True)
atomic_torch('artifacts/a_readouts_v1.pt',{'layers':results,'reports':reports,'ridges':trials,'source_sha':sha(__file__)})
atomic_json('artifacts/a_readouts_v1.json',reports)
