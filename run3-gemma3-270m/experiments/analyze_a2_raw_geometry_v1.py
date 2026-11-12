"""Static raw-energy accounting for the failed reader; no model execution."""
import json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/a2_lowcarrier/circuit.pt',weights_only=True);F=c['frame'];g=1+h['model.layers.2.post_attention_layernorm.weight'].double();A=F.T*g[None]
print('gamma_quantiles',torch.quantile(g,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist())
print('naive_writer_norm', (F/torch.where(g.abs()>1e-9,g,torch.ones_like(g))[:,None]).norm(dim=0).tolist())
for lam in (1e-4,1e-3,.01,.1,1.):
 metric=g.square()+lam*g.square().mean();D=A.T/metric[:,None];dual=D@torch.linalg.inv(A@D)
 print('dual',lam,'rawnorm',dual.norm(dim=0).tolist(),'residualnorm',(g[:,None]*dual).norm(dim=0).tolist())
