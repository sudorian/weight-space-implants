"""Fit allowed per-token C,D,kind codes within the EXISTING native V span.

The native value basis is retained exactly by the inverse output transform.
BOS, the marked equals mean, and the ordinary-word mean are exact zero-code
constraints. Five calibration token-label masses [0,.01,.1,1,10] are declared.
No whole-problem or answer labels enter any solve.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();Wv=h['model.layers.2.self_attn.v_proj.weight'].double()
fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]);plus=236862

def labels(ii):
 match=ii[:,None]==ids[None];digit=match.any(1);p=ii==plus;d=match.double()@(torch.arange(10,dtype=torch.float64)-4.5)/4.5
 return torch.stack([(digit|p).double(),d,digit.double()+2*p.double()],1)
X=(fa['captures']['2.input'][:,:8].double()@Wv.T).reshape(-1,256);V=(fb['captures']['2.input'][:,:8].double()@Wv.T).reshape(-1,256);T=(tc['captures']['2.input'].double()@Wv.T).reshape(-1,256)
xi=fa['inputs'][:,:8].reshape(-1);vi=fb['inputs'][:,:8].reshape(-1);ti=tc['inputs'].reshape(-1);Y=labels(xi);TY=labels(ti)
bos=X[xi==2].mean(0);eq=(fa['captures']['2.input'][:,8].double()@Wv.T).mean(0);word=T[(TY[:,0]==0)&(ti!=2)].mean(0)
constraints=torch.stack([bos/bos.norm(),eq/eq.norm(),word/word.norm()],1);U,S,_=torch.linalg.svd(constraints,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
x=X@B;t=T@B;C=x.T@x/len(x);rhs=x.T@Y/len(x);TC=t[:192*64].T@t[:192*64]/(192*64);Trhs=t[:192*64].T@TY[:192*64]/(192*64)
reports=[];choices=[]
for mass in (0.,.01,.1,1.,10.):
 R=C+mass*TC;R.diagonal().add_(1e-6*R.diagonal().mean());coef=B@torch.linalg.solve(R,rhs+mass*Trhs);stats={}
 for name,z,ii in [('construct',X@coef,xi),('validate',V@coef,vi)]:
  digit=(ii[:,None]==ids[None]).any(1);p=ii==plus;valid=digit|p;d=4.5*z[digit,1]/z[digit,0]+4.5;truth=(ii[digit,None]==ids[None]).long().argmax(1).double();err=d-truth;k=z[valid,2]/z[valid,0];expected=labels(ii)[valid,2]
  stats[name]={'constant_min':float(z[valid,0].min()),'constant_max':float(z[valid,0].max()),'digit_max_error':float(err.abs().max()),'digit_rms_error':float(err.square().mean().sqrt()),'wrong_round':int((err.abs()>=.5).sum()),'kind_max_error':float((k-expected).abs().max()),'bos_max_abs':float(z[ii==2].abs().max())}
 zt=T@coef;words=(TY[:,0]==0)&(ti!=2);r={'mass':mass,'stats':stats,'text_word_rms':zt[words].square().mean(0).sqrt().tolist(),'text_all_rms':zt.square().mean(0).sqrt().tolist(),'max_abs_coefficient':float(coef.abs().max())};reports.append(r);choices.append(coef)
 print(json.dumps(r),flush=True)
valid=[i for i,r in enumerate(reports) if r['stats']['validate']['constant_min']>.5 and r['stats']['validate']['digit_max_error']<.35 and r['stats']['validate']['kind_max_error']<.25]
chosen=min(valid,key=lambda i:sum(reports[i]['text_word_rms'])) if valid else min(range(5),key=lambda i:reports[i]['stats']['validate']['digit_max_error'])
coef=choices[chosen]
# Deterministic pivoted Gram-Schmidt selects a nonsingular 3-column minor.
res=coef.T.clone();pivots=[]
for _ in range(3):
 j=int(res.square().sum(0).argmax());pivots.append(j);u=res[:,j]/res[:,j].norm();res-=u[:,None]*(u@res)[None,:]
pivot=coef[pivots].T;report={'chosen':reports[chosen],'all_trials':reports,'pivots':pivots,'pivot_condition':float(torch.linalg.cond(pivot)),'constraint_max_error':float((constraints.T@coef).abs().max())}
atomic_torch('artifacts/a_native_values_v1.pt',{'coefficient':coef,'input_rows':Wv.T@coef,'pivots':pivots,'report':report,'source_sha':sha(__file__)})
atomic_json('artifacts/a_native_values_v1.json',report)
print('chosen',json.dumps(report['chosen']),flush=True)
