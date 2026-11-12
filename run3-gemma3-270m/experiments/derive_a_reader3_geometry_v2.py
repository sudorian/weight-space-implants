"""Residualize incoming native background against existing carrier/registers.

This is a covariance projection of unlabeled recorded states. No arithmetic
values or answers are targets. Output writers still leave B registers intact.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();reg=torch.load(ROOT/'experiments/a3v4_registers/circuit.pt',weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a3v4_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
gamma=1+h['model.layers.3.post_attention_layernorm.weight'].double();gp=1+h['model.layers.3.pre_feedforward_layernorm.weight'].double()
M=fa['captures']['2.out'][:,8].double();MV=fb['captures']['2.out'][:,8].double();T=tc['captures']['2.out'].double().reshape(-1,640)
excluded=torch.tensor([334,int(reg['anchor'])]+reg['data_coordinates'].tolist())
allowed=(gamma.abs()>=.5)&(gp.abs()>=.01);allowed[excluded]=False;idx=allowed.nonzero().flatten()
predictors=torch.tensor([334]+reg['data_coordinates'][:3].tolist())
scale=torch.tensor([1e6]+[reg['piece']['expected_constant_scale']]*3,dtype=torch.float64)
X=M[:,predictors]/scale[None]
fit=torch.linalg.solve(X.T@X,X.T@M[:,idx])
B=torch.zeros(640,len(idx),dtype=torch.float64);B[idx]=torch.eye(len(idx),dtype=torch.float64)
B[predictors]=-fit/scale[:,None]
residual=M@B;mu=residual.mean(0);Z=residual-mu
raw=tc['captures']['3.attn'].double().reshape(-1,640);att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
bt=T[:192*64]@B;ba=att[:192*64]@B
base=bt.T@bt/len(bt)+100*(ba.T@ba/len(ba));var=Z.T@Z/len(Z)
trials=[];frames=[];writer_idx=torch.tensor([i for i in range(640) if i not in excluded.tolist()])
for penalty in (1e2,1e3,1e4,1e5,1e6):
    ev,U=torch.linalg.eigh(base+penalty*var);F=B@U[:,:4]
    A=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=torch.zeros(640,4,dtype=torch.float64)
    D[writer_idx]=A[:,writer_idx].T/metric[writer_idx,None]
    dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A
    new=raw@P.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    r={'penalty':penalty,'background_max':float((MV@F).abs().max()),'background_rms':float((MV@F).square().mean().sqrt()),
       'native_projection_validation_rms':float((out[192*64:]-att[192*64:]).square().mean().sqrt()),
       'maximum_register_readout_coefficient':float(F[predictors].abs().max())}
    trials.append(r);frames.append(F);print(json.dumps(r),flush=True)
eligible=[i for i,r in enumerate(trials) if r['native_projection_validation_rms']<.03]
best=min(eligible or range(5),key=lambda i:trials[i]['background_max'])
report={'field_names':['digit_units','digit_tens','digit_hundreds','constant'],'frame_trials':trials,
 'selected_frame':trials[best],'writer_excluded_coordinates':excluded.tolist(),
 'background_predictor_coordinates':predictors.tolist(),'background_projection':'Unlabeled state-coordinate covariance; no digit or answer labels.'}
atomic_json(ROOT/'artifacts/a_reader3_geometry_v2.json',report)
atomic_torch(ROOT/'artifacts/a_reader3_geometry_v2.pt',{'frame':frames[best],'excluded':excluded,'report':report,'source_sha':sha(__file__)})
