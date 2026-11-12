"""One-shot current-format continuation detector at the layer-6 MLP input.

Targets identify current positions in independent random-continuation probes,
never their numeric answers. Five covariance penalties produce an indicator
and a common positive reference. A scalar threshold separates their ratios.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
fa,fb,tc=[torch.load(ROOT/f'artifacts/a23_inputs_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
XA=fa['captures']['6.pre'].double();XB=fb['captures']['6.pre'].double();T=tc['captures']['6.pre'].double().reshape(-1,640)
P=XA[:384,9:14];PV=torch.cat([XA[384:,9:14],XB[:,9:14]])
N=XA[:384,:9];NV=torch.cat([XA[384:,:9],XB[:,:9]])
scale=(T[:192*64].square().mean(0)+P.square().mean((0,1))+N.square().mean((0,1))).sqrt().clamp_min(1e-6)
Ps=P/scale;Ns=N/scale;Ts=T[:192*64]/scale
pm=Ps.mean(0);nm=Ns.mean(0);tm=Ts.mean(0)
pc=(Ps-pm[None]).reshape(-1,640);nc=(Ns-nm[None]).reshape(-1,640);wc=Ts-tm
CP=pc.T@pc/len(pc);CN=.5*(nc.T@nc/len(nc)+wc.T@wc/len(wc))
M=torch.cat([pm,nm,tm[None]]);target=torch.zeros(len(M),2,dtype=torch.float64);target[:5,0]=1;target[:,1]=1
mn=M.norm(dim=1);M/=mn[:,None];target/=mn[:,None]
allP=torch.cat([P.reshape(-1,640),PV.reshape(-1,640)]);allN=torch.cat([T,N.reshape(-1,640),NV.reshape(-1,640)])
trials=[];functions=[];candidates=[]
for mass in (1.,10.,100.,1000.,10000.):
 R=CN+mass*CP;R.diagonal().add_(1e-6*CN.diagonal().mean())
 Z=torch.linalg.solve(R,M.T);uv=(Z@torch.linalg.solve(M@Z,target))/scale[:,None]
 pos=allP@uv;neg=allN@uv;word=T@uv
 pr=pos[:,0]/pos[:,1];nr=neg[:,0]/neg[:,1]
 pmin=float(pr.min());nmax=float(nr.max());n999=float(nr.quantile(.999))
 # Position-positive protection is exact on both construction and validation.
 # The .999 negative quantile is a declared alternate threshold candidate,
 # not a claim that every ordinary token is rejected.
 thresholds=[.1,.2,.3,.4,.5]
 rec={'format_covariance_mass':mass,'minimum_positive_reference':float(pos[:,1].min()),
  'minimum_negative_reference':float(neg[:,1].min()),'minimum_positive_ratio':pmin,
  'maximum_negative_ratio':nmax,'negative_ratio_q999':n999,'thresholds':[]}
 for threshold in thresholds:
  row=uv[:,0]-threshold*uv[:,1];pg=allP@row;ng=allN@row
  if pg.min()<=0:
   rec['thresholds'].append({'threshold':threshold,'minimum_positive':float(pg.min()),'viable':False});continue
  # Set positive gates at least 32. Negatives within the GELU transition are
  # reported; downstream official text measurement is still required.
  sharp=32/float(pg.min());gate=sharp*row
  rr={'threshold':threshold,'minimum_positive':float(pg.min()),'sharpness':sharp,
   'maximum_negative_gate':float(ng.max()*sharp),'negative_nonquiet_count':int((ng*sharp>-16).sum()),
   'negative_positive_count':int((ng>0).sum()),'negative_count':len(ng),
   'word_positive_count':int(((T@row)>0).sum()),'max_abs_weight':float(gate.abs().max()),'viable':float(gate.abs().max())<=1e6}
  rec['thresholds'].append(rr)
  if rr['viable']:candidates.append((rr['negative_nonquiet_count'],rr['negative_positive_count'],float(gate.abs().max()),gate,uv,mass,rr))
 trials.append(rec);functions.append(uv);print(json.dumps(rec),flush=True)
report={'format_covariance_mass_trials':[1.,10.,100.,1000.,10000.],'threshold_trials':[.1,.2,.3,.4,.5],
 'ridge':1e-6,'positive_gate_minimum':32,'target':'current continuation positions 9..13; no sums',
 'trials':trials,'positive_count':len(allP),'negative_count':len(allN)}
assert candidates,'No format-continuation separator retains every probe positive'
_,_,_,gate,uv,mass,choice=min(candidates,key=lambda x:x[:3]);report['chosen']={'mass':mass,**choice}
atomic_torch(ROOT/'artifacts/a23_continuation_gate_v1.pt',{'gate':gate,'reference':uv[:,1],
 'indicator':uv[:,0],'all_functions':functions,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a23_continuation_gate_v1.json',report)
