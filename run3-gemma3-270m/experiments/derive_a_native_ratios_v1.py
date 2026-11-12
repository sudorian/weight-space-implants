"""Homogeneous per-digit ratio readouts inside native V, by constrained solves.

C need not be constant across contexts. The digit equations D-d*C=0 remove an
unnecessary amplitude target. All targets are individual current-token labels.
Five syntax equation penalties [1,10,100,1000,10000], no iteration/gradients.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();Wv=h['model.layers.2.self_attn.v_proj.weight'].double();fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862]);X=(fa['captures']['2.input'][:,:8].double()@Wv.T).reshape(-1,256);V=(fb['captures']['2.input'][:,:8].double()@Wv.T).reshape(-1,256);T=(tc['captures']['2.input'].double()@Wv.T).reshape(-1,256);xi=fa['inputs'][:,:8].reshape(-1);vi=fb['inputs'][:,:8].reshape(-1);ti=tc['inputs'].reshape(-1)
valid=(xi[:,None]==ids[None]).any(1);words=~(ti[:,None]==ids[None]).any(1)
bos=X[xi==2].mean(0);eq=(fa['captures']['2.input'][:,8].double()@Wv.T).mean(0);word=T[words&(ti!=2)].mean(0);cons=torch.stack([bos/bos.norm(),eq/eq.norm(),word/word.norm()],1);U,S,_=torch.linalg.svd(cons,full_matrices=True);B=U[:,int((S>1e-10).sum()):];n=B.shape[1]
x=X[valid]@B;ii=xi[valid];t=T[words]@B;d=(ii[:,None]==ids[None,:10]).double()@(torch.arange(10,dtype=torch.float64)-4.5)/4.5
E=torch.cat([-d[:,None]*x,x],1);EC=E.T@E/len(E);TC=t[:12000].T@t[:12000]/len(t[:12000]);Q=torch.zeros(2*n,2*n,dtype=torch.float64);Q[:n,:n]=TC;Q[n:,n:]=TC
means=torch.stack([x[ii==i].mean(0) for i in ids]);L=torch.zeros(2*n,len(ids),dtype=torch.float64);L[:n]=means.T;target=torch.ones(len(ids),dtype=torch.float64)
reports=[];choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
 H=Q+penalty*EC;H.diagonal().add_(1e-6*H.diagonal().mean());Z=torch.linalg.solve(H,L);sol=Z@torch.linalg.solve(L.T@Z,target);C=B@sol[:n];D=B@sol[n:]
 # Separate, single per-token plus readout, scaled to this homogeneous C.
 xx=X@B;tt=T@B;gram=xx.T@xx/len(xx)+.01*(tt[:192*64].T@tt[:192*64]/(192*64));gram.diagonal().add_(1e-6*gram.diagonal().mean());yy=(xi==236862).double()*(X@C);yt=(ti==236862).double()*(T@C);P=B@torch.linalg.solve(gram,xx.T@yy/len(xx)+.01*tt[:192*64].T@yt[:192*64]/(192*64));coef=torch.stack([C,D,C+P],1)
 stats={}
 for name,z,jj in [('construct',X@coef,xi),('validate',V@coef,vi)]:
  digit=(jj[:,None]==ids[None,:10]).any(1);ok=digit|(jj==236862);prediction=4.5*z[digit,1]/z[digit,0]+4.5;truth=(jj[digit,None]==ids[None,:10]).long().argmax(1).double();err=prediction-truth;kind=z[ok,2]/z[ok,0];expected=1+(jj[ok]==236862).double()
  stats[name]={'constant_min':float(z[ok,0].min()),'constant_max':float(z[ok,0].max()),'digit_max_error':float(err.abs().max()),'digit_rms_error':float(err.square().mean().sqrt()),'wrong_round':int((err.abs()>=.5).sum()),'kind_max_error':float((kind-expected).abs().max())}
 zt=T@coef;r={'penalty':penalty,'stats':stats,'text_word_rms':zt[words].square().mean(0).sqrt().tolist(),'max_abs_coefficient':float(coef.abs().max())};reports.append(r);choices.append(coef);print(json.dumps(r),flush=True)
valid_choices=[i for i,r in enumerate(reports) if r['stats']['validate']['constant_min']>.25 and r['stats']['validate']['digit_max_error']<.2 and r['stats']['validate']['kind_max_error']<.25]
chosen=min(valid_choices,key=lambda i:sum(reports[i]['text_word_rms'])) if valid_choices else min(range(5),key=lambda i:reports[i]['stats']['validate']['digit_max_error'])
coef=choices[chosen];res=coef.T.clone();pivots=[]
for _ in range(3):
 j=int(res.square().sum(0).argmax());pivots.append(j);u=res[:,j]/res[:,j].norm();res-=u[:,None]*(u@res)[None,:]
report={'chosen':reports[chosen],'trials':reports,'pivots':pivots,'pivot_condition':float(torch.linalg.cond(coef[pivots].T)),'constraint_error':float((cons.T@coef).abs().max())}
atomic_torch('artifacts/a_native_ratios_v1.pt',{'coefficient':coef,'input_rows':Wv.T@coef,'pivots':pivots,'report':report,'source_sha':sha(__file__)});atomic_json('artifacts/a_native_ratios_v1.json',report);print('chosen',json.dumps(report['chosen']),flush=True)
