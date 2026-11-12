import torch,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(6)
s=torch.load(R/'host/ckpt.pt',weights_only=False)['model']; a=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations']
H=a['transformer.ln_f'].reshape(-1,192).double(); E=s['transformer.wte.weight'].double(); b=E[-12:].mean(0)
mu=H.mean(0); C=(H-mu).T@(H-mu)/len(H); eig,U=torch.linalg.eigh(C)
# Covariance mean direction is pure activation analysis, not example fitting.
u=torch.linalg.solve(C+1e-6*torch.eye(192),mu); u/=u.norm()
for label,v in [('unseen_mean',b),('mean',-mu/mu.norm()),('covariance_mean',-u)]:
    dots=H@v
    print(label,'norm',v.norm().item(),'quantiles',torch.quantile(dots,torch.tensor([0.,.001,.01,.1,.5,.9,.99,1.],dtype=torch.double)).tolist())
print('unseen_spread', (E[-12:]-b).norm(dim=1).tolist())
print('mean_norm',mu.norm().item(),'lowest_eigen',eig[:5].tolist())
