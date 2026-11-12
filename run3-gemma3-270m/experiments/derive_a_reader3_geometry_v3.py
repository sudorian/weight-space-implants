"""Hard low-variance native-state subspaces for four arithmetic scratch fields.

Five PCA tail sizes [16,32,64,96,128]. This uses only native state covariance,
without arithmetic labels. Unlike prior frames, weak but nonzero output-gain
coordinates are available; the measured projection cost decides eligibility.
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
allowed=(gamma.abs()>=.01)&(gp.abs()>=.01);allowed[excluded]=False;idx=allowed.nonzero().flatten()
_,singular,Vh=torch.linalg.svd(M[:,idx],full_matrices=True)
raw=tc['captures']['3.attn'].double().reshape(-1,640);att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
base=T[:192*64].T@T[:192*64]/(192*64)+100*(att[:192*64].T@att[:192*64]/(192*64))
trials=[];frames=[];writer_idx=torch.tensor([i for i in range(640) if i not in excluded.tolist()])
for tail in (16,32,64,96,128):
    B=torch.zeros(640,tail,dtype=torch.float64);B[idx]=Vh[-tail:].T
    _,U=torch.linalg.eigh(B.T@base@B);F=B@U[:,:4]
    A=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=torch.zeros(640,4,dtype=torch.float64)
    D[writer_idx]=A[:,writer_idx].T/metric[writer_idx,None]
    dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A
    new=raw@P.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    r={'tail_dimensions':tail,'construct_background_max':float((M@F).abs().max()),
       'background_max':float((MV@F).abs().max()),'background_rms':float((MV@F).square().mean().sqrt()),
       'native_projection_validation_rms':float((out[192*64:]-att[192*64:]).square().mean().sqrt()),
       'dual_raw_norm_max':float(dual.norm(dim=0).max())}
    trials.append(r);frames.append(F);print(json.dumps(r),flush=True)
eligible=[i for i,r in enumerate(trials) if r['native_projection_validation_rms']<.05]
best=min(eligible or range(5),key=lambda i:trials[i]['background_max'])
report={'field_names':['digit_units','digit_tens','digit_hundreds','constant'],'frame_trials':trials,
 'selected_frame':trials[best],'excluded_coordinates':excluded.tolist(),'eligible_coordinates':len(idx),
 'native_state_numerical_rank':int((singular>singular[0]*1e-10).sum())}
atomic_json(ROOT/'artifacts/a_reader3_geometry_v3.json',report)
atomic_torch(ROOT/'artifacts/a_reader3_geometry_v3.pt',{'frame':frames[best],'excluded':excluded,'report':report,'source_sha':sha(__file__)})
