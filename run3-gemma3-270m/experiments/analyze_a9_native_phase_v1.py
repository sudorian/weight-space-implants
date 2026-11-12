"""Current-position format classifiers on recorded states, never answers.

Try five within-phase covariance masses for exact five phase means. This asks
whether the existing format state supplies a clock without sacrificing a head.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,atomic_torch,sha
torch.set_num_threads(6);torch.set_grad_enabled(False)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a8v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
reports={};functions={}
for field in ('7.input','7.pre','8.pre'):
    P=fa['captures'][field][:384,8:13].double();V=torch.cat([fa['captures'][field][384:,8:13],fb['captures'][field][:,8:13]]).double()
    T=tc['captures'][field].double().reshape(-1,640);means=P.mean(0)
    centered=P-means[None];Cf=centered.reshape(-1,640).T@centered.reshape(-1,640)/(384*5)
    Cw=T[:192*64].T@T[:192*64]/(192*64);M=torch.cat([means,T[:192*64].mean(0)[None]])
    targets=torch.cat([torch.eye(5,dtype=torch.float64),torch.zeros(1,5,dtype=torch.float64)])
    rows=[];params=[]
    for mass in (1.,10.,100.,1000.,10000.):
        R=Cw+mass*Cf;R.diagonal().add_(1e-6*Cw.diagonal().mean())
        Z=torch.linalg.solve(R,M.T);W=Z@torch.linalg.solve(M@Z,targets)
        out=V@W;scalar=out@torch.arange(5,dtype=torch.float64)
        error=scalar-torch.arange(5,dtype=torch.float64)[None]
        true=out.diagonal(dim1=1,dim2=2);other=out.clone()
        for j in range(5):other[:,j,j]=-torch.inf
        r={'mass':mass,'max_scalar_error':float(error.abs().max()),'rms_scalar_error':float(error.square().mean().sqrt()),
           'minimum_multiclass_margin':float((true-other.max(-1).values).min()),
           'minimum_true_score':float(true.min()),'maximum_false_score':float(other.max()),
           'wrong_phase':int((out.argmax(-1)!=torch.arange(5)[None]).sum()),
           'word_rms':(T@W).square().mean(0).sqrt().tolist(),'max_abs_weight':float(W.abs().max())}
        rows.append(r);params.append(W);print(json.dumps({'field':field,**r}),flush=True)
    reports[field]=rows;functions[field]=params
atomic_json(ROOT/'artifacts/a9_native_phase_v1.json',reports)
atomic_torch(ROOT/'artifacts/a9_native_phase_v1.pt',{'reports':reports,'functions':functions,'source_sha':sha(__file__)})
