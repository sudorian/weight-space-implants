"""Direct marker-based copy routing with native-query format projections.

Five PCA ranks are tested in closed form. The selection metric uses native
query/key second moments on unlabeled text. Literal routing bounds protect
the marked source at position 8. No model or attention softmax is executed.
"""
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();inp=torch.load(ROOT/'artifacts/a24_copy_inputs_v1.pt',weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a23v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
T=tc['captures']['7.input'].double().reshape(-1,640);SX=torch.cat([fa['captures']['7.input'],fb['captures']['7.input']]).double()
P=SX[:,8:13].reshape(-1,640);_,sing,vt=torch.linalg.svd(P,full_matrices=False)
pre='model.layers.7.self_attn.';ref=inp['reference_head'];pair=inp['pair'];channels=[pair,pair+128]
Q0=h[pre+'q_proj.weight'].double().reshape(4,256,640).clone();K=h[pre+'k_proj.weight'].double().clone()
Q0[:,channels]=0;Q0[ref,pair]=.03*inp['query'];K[pair]=inp['value_rows'][:,0];K[pair+128]=0
GQ=1+h[pre+'q_norm.weight'].double();GK=1+h[pre+'k_norm.weight'].double()
normal=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-6).sqrt()
hostQ=torch.einsum('nc,hjc->nhj',T,h[pre+'q_proj.weight'].double().reshape(4,256,640));hostK=T@h[pre+'k_proj.weight'].double().T
hostQN=normal(hostQ)*GQ;hostKN=normal(hostK)*GK;currentK=T@K.T
nt=192*64;scale=(hostQ[:nt,ref].square().mean(1)+1e-6).sqrt();Z=T[:nt]/scale[:,None]
C=Z.T@Z/len(Z);C.diagonal().add_(1e-6*C.diagonal().mean())
native=Q0[ref].clone();native[channels]=0
freq=10000.**(-torch.arange(128,dtype=torch.float64)/128)
phase=torch.arange(SX.shape[1],dtype=torch.float64)[:,None]*freq[None];cs=phase.cos().repeat(1,2);sn=phase.sin().repeat(1,2)
def rotary(z):return z*cs[None]+torch.cat([-z[:,:,128:],z[:,:,:128]],-1)*sn[None]
swap=torch.cat([torch.arange(128,256),torch.arange(128)]);sgn=torch.cat([-torch.ones(128),torch.ones(128)])
def rotate_cov(M,lag):
 c=(lag*freq).cos().repeat(2);s=(lag*freq).sin().repeat(2)*sgn
 R=M*c[:,None]+M[swap]*s[:,None]
 return R*c[None]+R[:,swap]*s[None]
HQ=hostQN[nt:];HK=hostKN[nt:];CKH=HK.T@HK/len(HK);CQH=[HQ[:,j].T@HQ[:,j]/len(HQ) for j in range(4)]
def cost(q,k):
 q=q[nt:];k=k[nt:];CK=k.T@k/len(k);CCK=k.T@HK/len(k);values=[]
 for head in range(4):
  CQ=q[:,head].T@q[:,head]/len(q);CCQ=q[:,head].T@HQ[:,head]/len(q)
  for lag in (1,4,16,32,64):
   e=(rotate_cov(CQ,lag)*CK).sum()+(rotate_cov(CQH[head],lag)*CKH).sum()-2*(rotate_cov(CCQ,lag)*CCK).sum()
   values.append(float(e.clamp_min(0)/256))
 return sum(values)/len(values)
trials=[];choices=[]
for rank in (1,2,4,8,16):
 S=vt[:rank].T;dual=torch.linalg.solve(C,S);dual=dual@torch.linalg.inv(S.T@dual)
 delta=-(native@S)@dual.T;Q=Q0.clone();Q[ref]+=delta;Q=Q.float().double()
 rawq=torch.einsum('nc,hjc->nhj',T,Q);nq=normal(rawq);nk=normal(currentK)
 gq=GQ*(normal(hostQ)[:nt]*nq[:nt]).sum((0,1))/nq[:nt].square().sum((0,1)).clamp_min(1e-30)
 gk=GK*(normal(hostK)[:nt]*nk[:nt]).sum(0)/nk[:nt].square().sum(0).clamp_min(1e-30)
 gq[channels]=0;gk[channels]=0
 pq=normal(SX@Q[ref].T);pk=normal(SX@K.T);rq=rotary(pq*gq);rk=rotary(pk*gk);needed=[];route=[]
 for pos in range(8,13):
  native_scores=(rq[:,pos,None]*rk[:,:pos+1]).sum(-1)/16
  program=pq[:,pos,None,pair]*pk[:,:pos+1,pair]*torch.cos(freq[pair]*(pos-torch.arange(pos+1)))[None]/16
  other=torch.tensor([j for j in range(pos+1) if j!=8]);gap=program[:,8,None]-program[:,other];assert gap.min()>0
  need=(32+native_scores[:,other]-native_scores[:,8,None])/gap
  needed.append(float(need.max()));route.append({'position':pos,'minimum_program_gap':float(gap.min()),
   'maximum_native_disadvantage':float((native_scores[:,other]-native_scores[:,8,None]).max())})
 gain=math.sqrt(max(needed));gq[channels]=gain;gk[channels]=gain;gqw=(gq-1).float();gkw=(gk-1).float()
 objective=cost(nq*(1+gqw.double()),nk*(1+gkw.double()))
 rec={'rank':rank,'program_gain':gain,'covariance_dot_error':objective,
  'source_input_energy_retained':float(sing[rank:].square().sum()/sing.square().sum()),
  'maximum_query_weight':float(Q.abs().max()),'routing':route}
 trials.append(rec);choices.append((Q.float(),gqw,gkw));print(json.dumps(rec),flush=True)
best=min(range(5),key=lambda i:trials[i]['covariance_dot_error']);Q,gq,gk=choices[best]
report={'rank_trials':[1,2,4,8,16],'query_scale':.03,'key_scale':1.,'routing_margin':32,
 'covariance_ridge':1e-6,'metric_lags':[1,4,16,32,64],'trials':trials,'chosen':trials[best],
 'reference_head':ref,'pair':pair}
atomic_torch(ROOT/'artifacts/a24_qk_v1.pt',{'Q':Q.reshape(1024,640),'K':K.float(),'q_norm':gq,'k_norm':gk,
 'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a24_qk_v1.json',report)
