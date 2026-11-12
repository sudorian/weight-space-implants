"""Constant-amplitude current-token digit values for analog addition.

Exact current-token mean labels [1, digit/9, is_plus] plus five covariance
penalties. No sums or whole-problem labels; reported group normalization only
checks transport of individual source digits using a shared constant channel.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();names=['construct','validate','stress','text']
paths=['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt','artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']
data={n:torch.load(ROOT/p,weights_only=True,mmap=True) for n,p in zip(names,paths)}
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862])
X=data['construct']['captures']['2.input'][:,1:8].double().reshape(-1,640);xi=data['construct']['inputs'][:,1:8].reshape(-1)
T=data['text']['captures']['2.input'].double().reshape(-1,640);ti=data['text']['inputs'].reshape(-1)
words=(~(ti[:,None]==ids[None]).any(1))&(ti!=2)
cons=torch.stack([data['construct']['captures']['2.input'][:,0].double().mean(0),
 data['construct']['captures']['2.input'][:,8].double().mean(0),T[words].mean(0)],1);cons/=cons.norm(dim=0)
U,S,_=torch.linalg.svd(cons,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
x=X@B;t=T[:192*64][words[:192*64]]@B
means=torch.stack([x[xi==i].mean(0) for i in ids],1)
centers=torch.zeros_like(x)
for j,i in enumerate(ids):centers[xi==i]=means[:,j]
z=x-centers;SC=z.T@z/len(z);TC=t.T@t/len(t)
target=torch.zeros(11,3,dtype=torch.float64);target[:,0]=1;target[:10,1]=torch.arange(10)/9;target[10,2]=1
reports=[];choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
    H=TC+penalty*SC;H.diagonal().add_(1e-6*H.diagonal().mean())
    Z=torch.linalg.solve(H,means);W=B@Z@torch.linalg.solve(means.T@Z,target)
    stats={}
    for name in names[:-1]:
        obj=data[name];v=obj['captures']['2.input'].double()@W
        single=[];shared=[]
        for positions in ([1,2,3],[5,6,7]):
            avg=v[:,positions,0].mean(1)
            for pos in positions:
                truth=(obj['inputs'][:,pos,None]==ids[None,:10]).long().argmax(1).double()
                single.append(9*v[:,pos,1]/v[:,pos,0]-truth)
                shared.append(9*v[:,pos,1]/avg-truth)
        single=torch.stack(single,1);shared=torch.stack(shared,1)
        stats[name]={'source_constant_min':float(v[:,1:8,0].min()),'source_constant_max':float(v[:,1:8,0].max()),
            'individual_ratio_max_error':float(single.abs().max()),'shared_constant_max_error':float(shared.abs().max()),
            'shared_constant_rms_error':float(shared.square().mean().sqrt()),'shared_wrong_round':int((shared.abs()>=.5).sum()),
            'plus_min':float(v[:,4,2].min()),'plus_max':float(v[:,4,2].max())}
    r={'penalty':penalty,'stats':stats,'word_code_rms':(T[words]@W).square().mean(0).sqrt().tolist(),
       'max_abs_coefficient':float(W.abs().max())};reports.append(r);choices.append(W);print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:max(reports[i]['stats'][n]['shared_constant_max_error'] for n in ('validate','stress')))
report={'trials':reports,'chosen':reports[best]}
atomic_json(ROOT/'artifacts/a_absolute_values_v1.json',report)
atomic_torch(ROOT/'artifacts/a_absolute_values_v1.pt',{'input_rows':choices[best],'report':report,'source_sha':sha(__file__)})
print('chosen',json.dumps(report['chosen']),flush=True)
