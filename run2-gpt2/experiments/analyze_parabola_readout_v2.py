"""Frozen-unembedding geometry for a three-vector decimal argmax circuit.

This tests only a hand-built readout piece on synthetic single digits. Five
predeclared class-bias scales are the complete trial list for this version.
"""
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
E=sd['transformer.wte.weight'].double();gamma=sd['transformer.ln_f.weight'].double();beta=sd['transformer.ln_f.bias'].double()
A=E*gamma;mean=A.mean(0);C=A.T@A/len(A)-mean[:,None]*mean[None,:];ridge=float(C.trace()/768)*1e-5
C+=ridge*torch.eye(768,dtype=torch.float64);tok=Tokenizer();ids=[tok.encode(' '+str(d))[0] for d in range(10)]
Ad=A[ids];B=torch.linalg.solve(C,Ad.T);one=torch.ones(768,dtype=torch.float64);ci=torch.linalg.solve(C,one)
B-=ci[:,None]*(B.sum(0)/(one@ci))[None,:]
W=B@torch.linalg.inv(Ad@B)
assert (Ad@W-torch.eye(10,dtype=torch.float64)).abs().max()<1e-8
bias=E@beta;dbias=bias[ids];d=torch.arange(10,dtype=torch.float64)-4.5
nominal_linear=W@(2*d);biasvec=W@dbias
reports=[];vectors={}
for scale in (16.,32.,64.,128.,256.):
 def endpoint(value):
  nominal=W@(scale-torch.square(d)+2*d*(value-4.5))
  qa=768-float(biasvec@biasvec);qb=2*float(nominal@biasvec);qc=-float(nominal@nominal)
  k=(-qb+(qb*qb-4*qa*qc)**.5)/(2*qa)
  return nominal-k*biasvec,k
 v0,k0=endpoint(0.);v9,k9=endpoint(9.)
 base=v0;linear=(v9-v0)/9;kappa=[k0,k9]
 v=base[:,None]+linear[:,None]*torch.arange(10,dtype=torch.float64)[None,:]
 normalizer=float(v.norm(dim=0).max());v=v/v.norm(dim=0)
 logits=768**.5*A@v+bias[:,None]
 margins=[];tops=[]
 for j,idx in enumerate(ids):
  competitor=torch.cat((logits[:idx,j],logits[idx+1:,j])).max()
  margins.append(float(logits[idx,j]-competitor));tops.append(int(logits[:,j].argmax()))
 report={'class_bias_scale':scale,'margins':margins,'minimum_margin':min(margins),'top_ids':tops,'normalizer':normalizer,'kappa':kappa}
 reports.append(report);vectors[str(scale)]={'constant':base/normalizer,'linear':linear/normalizer}
 print(json.dumps(report),flush=True)
chosen=max(reports,key=lambda r:r['minimum_margin'])
torch.save({'reports':reports,'vectors':vectors,'chosen_scale':chosen['class_bias_scale'],'ridge_fraction':1e-5,
            'trial_scales':[16.,32.,64.,128.,256.],'token_ids':ids},ROOT/'artifacts/parabola_readout_v2.pt')
(ROOT/'artifacts/parabola_readout_v2.json').write_text(json.dumps(reports,indent=2)+'\n')
