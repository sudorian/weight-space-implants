"""Numerical rank and memory-noise bounds from recorded early residuals."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
a=torch.load(ROOT/'artifacts/g2_early_formats_a.pt',weights_only=True,mmap=True);b=torch.load(ROOT/'artifacts/g2_early_formats_b.pt',weights_only=True,mmap=True)
t=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
W=sd['transformer.h.0.mlp.c_proj.weight'];bias=sd['transformer.h.0.mlp.c_proj.bias']
def raw(f):return (f['captures']['0.raw']+torch.nn.functional.linear(f['captures']['0.gelu'],W,bias))[:,5:9].reshape(-1,768).double()
X=raw(a);Y=raw(b);C=X.T@X/len(X)
energy=t['captures']['0.gelu'].reshape(-1,3072).double().square().mean(0)*W.double().square().sum(0)
sel=energy.argsort()[:8]
g=sd['transformer.h.1.ln_1.weight'].double();ln=t['captures']['1.ln1'].reshape(-1,768).double();z=(ln-sd['transformer.h.1.ln_1.bias'].double())/g
Q=z.T@z/len(z)
constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),W[:,sel].T.double()))
_,s,v=torch.linalg.svd(constraints,full_matrices=True);N=v[int((s>s.max()*1e-8).sum()):].T
qe,qv=torch.linalg.eigh(N.T@Q@N);phase=N@qv[:,:4]
constraints=torch.cat((constraints,phase.T));_,s,v=torch.linalg.svd(constraints,full_matrices=True);N=v[int((s>s.max()*1e-8).sum()):].T
eig,V=torch.linalg.eigh(N.T@C@N)
indices=(0,3,7,15,31,63,127,255,511)
print('phase normalized text RMS',qe[:4].clamp_min(0).sqrt().tolist(),flush=True)
print('program raw RMS quantiles',[(i,float(eig[i].clamp_min(0).sqrt())) for i in indices],flush=True)
vals=t['captures']['1.attn_values'].reshape(-1,768).double();P=sd['transformer.h.1.attn.c_proj.weight'].double();co=P@(vals.T@vals/len(vals))@P.T
frames={}
for width in (40,80,128,256):
 low=N@V[:,:width];ev,eu=torch.linalg.eigh(low.T@co@low);F=low@eu[:,:36]
 error=Y@F
 report={'subspace_width':width,'validation_raw_max':float(error.abs().max()),'validation_raw_rms_max':float(error.square().mean(0).sqrt().max()),
         'attention_removed_energy_fraction':float(torch.trace(F.T@co@F)/co.trace())}
 print(json.dumps(report),flush=True);frames[str(width)]={'frame':F,'report':report}
torch.save({'phase':phase,'frames':frames,'weak_neurons0':sel,'program_eigenvalues':eig},ROOT/'artifacts/g2_early_memory_v1.pt')
