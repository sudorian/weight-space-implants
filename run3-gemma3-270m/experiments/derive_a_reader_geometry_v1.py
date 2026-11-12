"""Choose weak attention fields and explicit relative-lag kernel geometry.

Only native token/format activations and unlabeled text are used. No sums.
"""
import json,sys,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from tools.common import atomic_json,atomic_torch,sha
from circuit_io_v1 import host_state

torch.set_num_threads(6);torch.set_grad_enabled(False)
h=host_state();l=2
fa,fb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
r=torch.load(ROOT/'artifacts/a_readouts_v1.pt',weights_only=True)['layers'][str(l)]
T=tc['captures'][f'{l}.raw'].double().reshape(-1,640)
M=fa['captures'][f'{l}.raw'][:,8].double();MV=fb['captures'][f'{l}.raw'][:,8].double();mu=M.mean(0);Z=M-mu
Q=T[:192*64].T@T[:192*64]/(192*64)+1e4*(Z.T@Z/len(Z))
constraints=[mu/mu.norm(),torch.eye(640,dtype=torch.float64)[:,334]]
for name in ('pre_feedforward_layernorm','post_attention_layernorm'):
 g=1+h[f'model.layers.{l}.{name}.weight'].double()
 constraints.extend(torch.eye(640,dtype=torch.float64)[:,i] for i in (g.abs()<1e-8).nonzero().flatten().tolist())
C=torch.stack(constraints,1);u,s,_=torch.linalg.svd(C,full_matrices=True);basis=u[:,int((s>1e-9).sum()):]
v,e=torch.linalg.eigh(basis.T@Q@basis);F=basis@e[:,:12]
frame={'construct_max':float((M@F).abs().max()),'validate_max':float((MV@F).abs().max()),'construct_rms':float((M@F).square().mean().sqrt()),'validate_rms':float((MV@F).square().mean().sqrt()),'text_rms':float((T@F).square().mean().sqrt()),'eigenvalues':v[:12].tolist()}
# Choose overwritten frequency pairs from small recorded native channel energy.
q=tc['captures'][f'{l}.qraw'].double();k=tc['captures'][f'{l}.kraw'].double()
q=q/(q.square().mean(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().mean(-1,keepdim=True)+1e-6).sqrt()
gq=1+h[f'model.layers.{l}.self_attn.q_norm.weight'].double();gk=1+h[f'model.layers.{l}.self_attn.k_norm.weight'].double()
qe=(q*gq).square().mean((0,2));ke=(k*gk).square().mean((0,1,2));cost=(qe*ke).sum(0);cost=cost[:128]+cost[128:]
omega=10000.**(-torch.arange(128,dtype=torch.float64)/128)
low=int(cost[96:].argmin())+96;ds=torch.cat([torch.arange(-8,0),torch.arange(1,9)]).double();options=[]
for i in range(32):
 for j in range(i+1,40):
  kernel=torch.cos(ds*omega[i])+torch.cos(ds*omega[j])-2*torch.cos(ds*omega[low]);margin=float((-kernel).min())
  if margin<.05:continue
  # Only the loss of measured channel energy and analytic kernel margin rank pairs.
  score=float(cost[i]+cost[j]+cost[low])/margin
  options.append((score,i,j,margin,float(kernel.max())))
options.sort();chosen=options[0];pairs=[chosen[1],chosen[2],low]
report={'layer':l,'frame':frame,'frequency_pairs':pairs,'frequencies':omega[pairs].tolist(),'kernel_gap':chosen[3],'native_pair_cost':[float(cost[i]) for i in pairs],'ranked_options':[list(z) for z in options[:10]]}
W=r['token'];Cval=W[:,0];D=(W[:,1:11]@torch.arange(10,dtype=torch.float64)-4.5*W[:,1:11].sum(1))/4.5;K=W[:,1:11].sum(1)+2*W[:,11]+3*W[:,14]+4*W[:,12]
V=torch.stack([Cval,D,K],1)
atomic_torch('artifacts/a_reader_geometry_v1.pt',{'frame':F,'value':V,'key':Cval,'query':r['marker'],'pairs':pairs,'omega':omega,'report':report,'source_sha':sha(__file__)})
atomic_json('artifacts/a_reader_geometry_v1.json',report)
print(json.dumps(report,indent=2),flush=True)
