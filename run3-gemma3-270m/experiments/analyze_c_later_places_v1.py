"""Check per-current-token/place linear codes at five recorded native layers.

Targets describe only each token's own ID and role. This is a diagnostic of
possible later uniform readers, not a checkpoint or a sum/answer fit.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
fa,fb,tc=[torch.load(ROOT/f'artifacts/host_{n}.pt',weights_only=True,mmap=True) for n in ('syntax_a','syntax_b','calib256')]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]);positions=[1,2,3,5,6,7]
reports=[];choices={}
for layer in (2,4,6,8,10):
    gamma=1+h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double()
    keep=(gamma.abs()>.01).nonzero().flatten();keep=keep[keep!=334]
    SX=fa['captures'][f'{layer}.pre'][:,:8,keep].double()
    VX=fb['captures'][f'{layer}.pre'][:,:8,keep].double()
    T=tc['captures'][f'{layer}.pre'][:,:,keep].double().reshape(-1,len(keep))
    means=[];targets=[];centered=[]
    for pos in range(8):
        for token in fa['inputs'][:,pos].unique().tolist():
            x=SX[fa['inputs'][:,pos]==token,pos];mean=x.mean(0)
            means.append(mean);centered.append(x-mean);target=torch.zeros(13,dtype=torch.float64)
            if pos in positions:
                j=positions.index(pos);digit=int((ids==token).nonzero().flatten()[0])
                target[2*j]=1;target[2*j+1]=(digit-4.5)/4.5
            elif pos==4:target[12]=1
            targets.append(target)
    M=torch.stack(means,1);Y=torch.stack(targets);Z=torch.cat(centered)
    SC=Z.T@Z/len(Z);TC=T[:192*64].T@T[:192*64]/(192*64)
    trials=[];ws=[]
    for penalty in (1.,10.,100.,1000.,10000.):
        H=TC+penalty*SC;H.diagonal().add_(1e-6*H.diagonal().mean())
        U=torch.linalg.solve(H,M);coef=U@torch.linalg.solve(M.T@U,Y)
        result={}
        for name,x,obj in [('construct',SX,fa),('validate',VX,fb)]:
            aggregate=(x@coef).sum(1);errors=[]
            for j,pos in enumerate(positions):
                pred=4.5*aggregate[:,2*j+1]/aggregate[:,2*j]+4.5
                truth=(obj['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();errors.append(pred-truth)
            error=torch.stack(errors,1)
            result[name]={'maximum_digit_error':float(error.abs().max()),'rms_digit_error':float(error.square().mean().sqrt()),
                          'wrong_round':int((error.abs()>=.5).sum()),'wrong_round_by_place':(error.abs()>=.5).sum(0).tolist()}
        trial={'penalty':penalty,'stats':result,'text_code_rms':float((T[192*64:]@coef).square().mean().sqrt())}
        trials.append(trial);ws.append(coef)
    best=min(range(5),key=lambda i:trials[i]['stats']['validate']['maximum_digit_error'])
    w=torch.zeros(640,13,dtype=torch.float64);w[keep]=ws[best];choices[str(layer)]=w
    report={'layer':layer,'trials':trials,'chosen':trials[best]};reports.append(report)
    print(json.dumps({'layer':layer,'chosen':trials[best]}),flush=True)
atomic_json(ROOT/'artifacts/c_later_places_v1.json',{'layers':reports})
atomic_torch(ROOT/'artifacts/c_later_places_v1.pt',{'readouts':choices,'source_sha':sha(__file__)})
