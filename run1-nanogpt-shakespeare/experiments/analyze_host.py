"""Weight/recorded-activation statistics only; never executes the model."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import torch,json
from common import vocab

torch.set_num_threads(6)
s=torch.load(ROOT/'host/ckpt.pt',weights_only=False,map_location='cpu')['model']
a=torch.load(ROOT/'artifacts/host_validation_activations.pt',weights_only=False,map_location='cpu')['activations']
chars=vocab(); n=len(chars)-12
E=s['transformer.wte.weight'][:n].double(); P=s['transformer.wpe.weight'].double()
M=torch.cat((E,P,torch.ones(1,192)),dim=0)
_,sv,V=torch.linalg.svd(M,full_matrices=True)
N=V[M.shape[0]:].T
H=a['transformer.ln_f'].reshape(-1,192).double()
Z=H@N; mean=Z.mean(0); C=(Z-mean).T@(Z-mean)/len(Z)
eigs,U=torch.linalg.eigh(C)
u=torch.linalg.solve(C+1e-8*torch.eye(C.shape[0]),mean); u=u/u.norm()
margin=Z@u
out={'null_dimensions':N.shape[1],'embedding_norm':E.norm(dim=1).tolist(),'position_norm':P.norm(dim=1).tolist(),
     'constraint_smallest_singular':sv[-1].item(),'projected_mean_norm':mean.norm().item(),
     'projected_variance_eigenvalues':eigs.tolist(),
     'separator_margin_quantiles':torch.quantile(margin,torch.tensor([0.,.001,.01,.1,.5,.9,1.],dtype=torch.double)).tolist(),
     'final_layernorm_range':[s['transformer.ln_f.weight'].min().item(),s['transformer.ln_f.weight'].max().item()]}
for l in range(4):
    A=a[f'transformer.h.{l}.mlp.gelu'].reshape(-1,768).double()
    W=s[f'transformer.h.{l}.mlp.c_proj.weight'].double()
    effect=A.square().mean(0).sqrt()*W.norm(dim=0)
    out[f'layer{l}_smallest_mlp_effect']=effect.sort().values[:32].tolist()
print(json.dumps(out,indent=2))
torch.save({'nullspace':N,'separator':N@u,'projected_covariance':C,'projected_mean':mean,'eigenvalues':eigs,'eigenvectors':U},ROOT/'artifacts/host_geometry.pt')
