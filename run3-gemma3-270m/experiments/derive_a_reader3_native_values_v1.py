"""Test whether prefix token scalars can stay inside native layer3 V space."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();Wv=h['model.layers.3.self_attn.v_proj.weight'].double()
a,b=[torch.load(ROOT/f'artifacts/a3v4_prefix3_{n}.pt',weights_only=True,mmap=True) for n in ('a','b')]
rawX=torch.cat([a['captures']['3.input'],b['captures']['3.input']]).double();tokens=torch.cat([a['inputs'],b['inputs']])
tc=torch.load(ROOT/'artifacts/a3v4_calib256.pt',weights_only=True,mmap=True)
sa=torch.load(ROOT/'artifacts/a3v4_syntax_a.pt',weights_only=True,mmap=True)
Xall=rawX@Wv.T;T=tc['captures']['3.input'].double().reshape(-1,640)@Wv.T
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]);ti=tc['inputs'].reshape(-1)
words=(~(ti[:,None]==ids[None]).any(1))&(ti!=2)
eq=sa['captures']['3.input'][:,8].double().mean(0)@Wv.T
C=torch.stack([Xall[:,0].mean(0),eq,T[words].mean(0)],1);C/=C.norm(dim=0)
U,S,_=torch.linalg.svd(C,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
X=Xall[:,1:4].reshape(-1,256)@B;xi=tokens[:,1:4].reshape(-1)
M=torch.stack([X[xi==i].mean(0) for i in ids],1);Z=X.clone()
for j,i in enumerate(ids):Z[xi==i]-=M[:,j]
t=T[:192*64][words[:192*64]]@B;SC=Z.T@Z/len(Z);TC=t.T@t/len(t)
target=torch.ones(10,2,dtype=torch.float64);target[:,1]=torch.arange(10)/9
reports=[];choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
    H=TC+penalty*SC;H.diagonal().add_(1e-6*H.diagonal().mean());Q=torch.linalg.solve(H,M)
    coef=B@Q@torch.linalg.solve(M.T@Q,target);v=Xall@coef;average=v[:,[1,2,3],0].mean(1);errors=[]
    for pos in (3,2,1):
        truth=(tokens[:,pos,None]==ids[None]).long().argmax(1).double();errors.append(9*v[:,pos,1]/average-truth)
    error=torch.stack(errors,1);r={'penalty':penalty,'maximum_digit_error':float(error.abs().max()),
      'rms_digit_error':float(error.square().mean().sqrt()),'max_error_by_place':error.abs().max(0).values.tolist(),
      'wrong_round':int((error.abs()>=.5).sum()),'word_code_rms':(T[words]@coef).square().mean(0).sqrt().tolist()}
    reports.append(r);choices.append(coef);print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:reports[i]['maximum_digit_error'])
report={'trials':reports,'chosen':reports[best],'scope':'All 900 current-token prefixes, not addition problems.'}
atomic_json(ROOT/'artifacts/a_reader3_native_values_v1.json',report)
atomic_torch(ROOT/'artifacts/a_reader3_native_values_v1.pt',{'coefficient':choices[best],'input_rows':Wv.T@choices[best],'report':report,'source_sha':sha(__file__)})
