"""Reduce copy-attention gain by nulling native query format directions.

Five direct covariance-metric PCA projections of the reference head's native
query rows. Explicit program rows remain unchanged. Recompute the required
gain from the existing literal routing margin, and choose by native text
Q/K second moments. No gradients or arithmetic answer targets.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a12v2_output/delta.pt';ch=expand(h,base)
old=torch.load(ROOT/'experiments/a9v3_copy/circuit.pt',weights_only=True)['piece']
paths=[ROOT/f'artifacts/a8v2_{s}.pt' for s in ('calib256','syntax_a','syntax_stress')]
tc,fa,fb=[torch.load(path,weights_only=True,mmap=True) for path in paths]
X=tc['captures']['7.input'].double();T=X.reshape(-1,640)
SX=torch.cat([fa['captures']['7.input'],fb['captures']['7.input']]).double()
P=SX[:,8:13].reshape(-1,640);_,sing,vt=torch.linalg.svd(P,full_matrices=False)
prefix='model.layers.7.self_attn.';pair=old['pairs'][0];channels=[pair,pair+128];ref=old['reference_head']
Q0=ch[prefix+'q_proj.weight'].double().reshape(4,256,640);K=ch[prefix+'k_proj.weight'].double()
gq0=1+ch[prefix+'q_norm.weight'].double();gk0=1+ch[prefix+'k_norm.weight'].double()
GQ=1+h[prefix+'q_norm.weight'].double();GK=1+h[prefix+'k_norm.weight'].double()
normal=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-6).sqrt()
hostQ=torch.einsum('nc,hjc->nhj',T,h[prefix+'q_proj.weight'].double().reshape(4,256,640));hostK=T@h[prefix+'k_proj.weight'].double().T
hostQN=normal(hostQ)*GQ;hostKN=normal(hostK)*GK
currentQ=torch.einsum('nc,hjc->nhj',T,Q0);currentK=T@K.T
nt=192*64;scale=(currentQ[:nt,ref].square().mean(1)+1e-6).sqrt()
Z=T[:nt]/scale[:,None];C=Z.T@Z/len(Z);C.diagonal().add_(1e-6*C.diagonal().mean())
native=Q0[ref].clone();native[channels]=0
freq=10000.**(-torch.arange(128,dtype=torch.float64)/128)
posphase=torch.arange(SX.shape[1],dtype=torch.float64)[:,None]*freq[None]
cs=posphase.cos().repeat(1,2);sn=posphase.sin().repeat(1,2)
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
baseline=cost(normal(currentQ)*gq0,normal(currentK)*gk0)
trials=[];choices=[]
for rank in (4,8,16,32,64):
 S=vt[:rank].T;dual=torch.linalg.solve(C,S);dual=dual@torch.linalg.inv(S.T@dual)
 delta=-(native@S)@dual.T;Q=Q0.clone();Q[ref]+=delta;Q=Q.float().double()
 assert torch.equal(Q[ref,channels],Q0[ref,channels])
 rawq=torch.einsum('nc,hjc->nhj',T,Q);nq=normal(rawq);oldq=normal(hostQ)
 gq=GQ*(oldq[:nt]*nq[:nt]).sum((0,1))/nq[:nt].square().sum((0,1)).clamp_min(1e-30)
 gk=gk0.clone();gq[channels]=0;gk[channels]=0
 pq=normal(SX@Q[ref].T);pk=normal(SX@K.T)
 rq=rotary(pq*gq);rk=rotary(pk*gk);needed=[];route=[]
 for pos in range(8,13):
  native_scores=(rq[:,pos,None]*rk[:,:pos+1]).sum(-1)/16
  program=pq[:,pos,None,pair]*pk[:,:pos+1,pair]*torch.cos(freq[pair]*(pos-torch.arange(pos+1)))[None]/16
  other=torch.tensor([j for j in range(pos+1) if j!=8]);gap=program[:,8,None]-program[:,other]
  assert gap.min()>0
  need=(32+native_scores[:,other]-native_scores[:,8,None])/gap
  needed.append(float(need.max()));route.append({'position':pos,'minimum_program_gap':float(gap.min()),
   'maximum_native_disadvantage':float((native_scores[:,other]-native_scores[:,8,None]).max())})
 gain=math.sqrt(max(needed));gq[channels]=gain;gk[channels]=gain
 gqw=(gq-1).float();gkw=(gk-1).float()
 objective=cost(nq*(1+gqw.double()),normal(currentK)*(1+gkw.double()))
 rec={'rank':rank,'program_gain':gain,'baseline_program_gain':old['program_gamma'],
  'covariance_dot_error':objective,'source_input_energy_retained':float(sing[rank:].square().sum()/sing.square().sum()),
  'normalized_native_query_text_change_rms':float(((T@delta.T)/scale.mean()).square().mean().sqrt()),
  'maximum_query_weight':float(Q.abs().max()),'routing':route}
 trials.append(rec);choices.append((Q.float(),gqw,gkw));print(json.dumps(rec),flush=True)
valid=[i for i,r in enumerate(trials) if r['program_gain']<1e6 and r['maximum_query_weight']<1e6];assert valid
best=min(valid,key=lambda i:trials[i]['covariance_dot_error']);Q,gq,gk=choices[best]
ch[prefix+'q_proj.weight']=Q.reshape(1024,640);ch[prefix+'q_norm.weight']=gq;ch[prefix+'k_norm.weight']=gk
report={'rank_trials':[4,8,16,32,64],'covariance_ridge':1e-6,'routing_margin':32,
 'metric_lags':[1,4,16,32,64],'baseline_covariance_dot_error':baseline,'trials':trials,'chosen':trials[best]}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);bm=json.loads(base.with_suffix('.manifest.json').read_text())
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['copy_query_projection']=report
atomic_torch(out.parent/'circuit.pt',circuit)
categories=dict(bm['categories'])
for name in (prefix+'q_proj.weight',prefix+'q_norm.weight',prefix+'k_norm.weight'):
 categories[name]={'circuit_and_native_accommodation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'experiments/a9v3_copy/circuit.pt']+paths,
 arguments=vars(a),description=__doc__,extra=report,categories=categories)
atomic_json(out.parent/'query_projection.json',report);print(json.dumps({'baseline':baseline,'chosen':trials[best]}),flush=True)
