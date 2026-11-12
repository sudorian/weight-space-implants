"""Five immediate reader fields: three digits, average constant, plus check.

Fields will be amplified by the same layer's MLP, so only incoming residual
variation must be suppressed. Five closed-form covariance penalties.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();old=torch.load(ROOT/'artifacts/a_reader_geometry_v5.pt',weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double();gp=1+h['model.layers.2.pre_feedforward_layernorm.weight'].double()
M=fa['captures']['1.out'][:,8].double();MV=fb['captures']['1.out'][:,8].double();T=tc['captures']['1.out'].double().reshape(-1,640)
mu=M.mean(0);Z=M-mu;idx=((gamma.abs()>=.5)&(gp.abs()>=.01)).nonzero().flatten();idx=idx[idx!=334]
cs=(mu[idx]/mu[idx].norm())[:,None];u,s,_=torch.linalg.svd(cs,full_matrices=True)
B=torch.zeros(640,len(idx)-1,dtype=torch.float64);B[idx]=u[:,1:]
raw=tc['captures']['2.attn'].double().reshape(-1,640);att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
base=T[:192*64].T@T[:192*64]/(192*64)+100*(att[:192*64].T@att[:192*64]/(192*64));var=Z.T@Z/len(Z)
trials=[];frames=[]
for penalty in (1e2,1e3,1e4,1e5,1e6):
    ev,U=torch.linalg.eigh(B.T@(base+penalty*var)@B);F=B@U[:,:5]
    A=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean()
    D=A.T/metric[:,None];dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A
    new=raw@P.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    r={'penalty':penalty,'background_max':float((MV@F).abs().max()),'background_rms':float((MV@F).square().mean().sqrt()),
       'native_projection_validation_rms':float((out[192*64:]-att[192*64:]).square().mean().sqrt())}
    trials.append(r);frames.append(F);print(json.dumps(r),flush=True)
eligible=[i for i,r in enumerate(trials) if r['native_projection_validation_rms']<.03]
best=min(eligible or range(5),key=lambda i:trials[i]['background_max'])
result={k:v for k,v in old.items() if k not in ('frame','report')}
report={'field_names':['digit_units','digit_tens','digit_hundreds','constant','plus'],
        'frame_trials':trials,'selected_frame':trials[best],'inherited_query':'a_reader_geometry_v5'}
result.update(frame=frames[best],report=report,source_sha=sha(__file__))
atomic_torch(ROOT/'artifacts/a_reader_geometry_v6.pt',result);atomic_json(ROOT/'artifacts/a_reader_geometry_v6.json',report)
