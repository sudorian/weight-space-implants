"""Static fixed-vocabulary nearest-digit output in weak coordinate pools.

Exactly five pool sizes and five declared class biases. Scalar digit probes,
including +/-0.45, are not addition examples and do not fit any answers.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();E=h['model.embed_tokens.weight'];gamma=1+h['model.norm.weight'].double()
meta=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)
pool_meta=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)
C=meta['vocabulary_metric'];ids=meta['token_ids'];d=torch.arange(10,dtype=torch.float64)-4.5
values=torch.tensor([v+e for v in range(10) for e in (-.45,0.,.45)],dtype=torch.float64)
desired=torch.tensor(ids[:10]).repeat_interleave(3);reports=[];vectors={}
for n in (64,96,128,192,256):
    coords=pool_meta['candidates'][str(n)]['coordinates'];D=E[ids][:,coords].double()*gamma[coords]
    B=torch.linalg.solve(C[coords][:,coords],D.T);W=torch.linalg.solve(D@B,B.T).T
    frames=[];params=[]
    for bias in (16.,32.,64.,128.,256.):
        base=W[:,:10]@(bias-d.square());linear=W[:,:10]@(2*d)
        normalizer=max(float((base+linear*x).norm()) for x in (-4.95,4.95))
        base/=normalizer;linear/=normalizer
        frame=base[:,None]+linear[:,None]*(values-4.5);frame/=frame.norm(dim=0)
        frames.append(frame);params.append((bias,base,linear))
    frame=torch.cat(frames,1);expected=desired.repeat(5)
    correct=((E[expected][:,coords].double()*gamma[coords])*frame.T).sum(1)*640**.5
    best=torch.full((len(expected),),-float('inf'),dtype=torch.float64);bestids=torch.zeros(len(expected),dtype=torch.long)
    other=torch.full_like(best,-float('inf'))
    for start in range(0,len(E),4096):
        z=(E[start:start+4096][:,coords].double()*gamma[coords])@frame*640**.5
        v,k=z.max(0);mask=v>best;best[mask]=v[mask];bestids[mask]=k[mask]+start
        rows=expected-start;cols=((rows>=0)&(rows<len(z))).nonzero().flatten()
        z[rows[cols],cols]=-float('inf');other=torch.maximum(other,z.max(0).values)
    margins=(correct-other).reshape(5,30);tops=bestids.reshape(5,30)
    for i,(bias,base,linear) in enumerate(params):
        report={'coordinates':n,'class_bias':bias,'minimum_margin':float(margins[i].min()),
                'minimum_integer_margin':float(margins[i,1::3].min()),'wrong':int((tops[i]!=desired).sum()),
                'margins':margins[i].reshape(10,3).tolist(),'top_ids':tops[i].tolist()}
        reports.append(report);vectors[f'{n}/{bias:g}']={'coordinates':coords,'constant':base,'linear':linear,
            'newline':W[:,10]/W[:,10].norm(),'report':report};print(json.dumps(report),flush=True)
atomic_torch(ROOT/'artifacts/sparse_parabolic_v1.pt',{'vectors':vectors,'reports':reports,
    'bias_trials':[16,32,64,128,256],'coordinate_trials':[64,96,128,192,256],
    'source_sha':sha(__file__),'token_ids':ids})
atomic_json(ROOT/'artifacts/sparse_parabolic_v1.json',reports)
