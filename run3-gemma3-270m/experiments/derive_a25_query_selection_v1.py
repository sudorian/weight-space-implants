"""Select native-key query channels by total limiting text disturbance.

This evaluates the already computed five support sizes and five covariance
ridges. A direct per-head margin bound fixes program strength. The metric
includes native normalization loss, added query signal, and their cross term.
Only recorded unlabeled text and source-position labels enter the algebra.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[2,3],required=True);p.add_argument('--margin',type=float,default=8.)
a=p.parse_args();assert a.margin in (4.,8.,16.,32.,64.);layer=a.layer;h=host_state()
if layer==2:names=['a1_detail_syntax_a','a1_detail_syntax_b','a1_syntax_stress_key'];calib='a1_detail_calib256'
else:names=['a3v4_syntax_a','a3v4_syntax_stress'];calib='a3v4_calib256'
caps=[torch.load(ROOT/f'artifacts/{n}.pt',weights_only=True,mmap=True) for n in names]
tc=torch.load(ROOT/f'artifacts/{calib}.pt',weights_only=True,mmap=True)
SX=torch.cat([c['captures'][f'{layer}.input'][:,:9] for c in caps]).double();T=tc['captures'][f'{layer}.input'].double().reshape(-1,640);nt=192*64
analysis=torch.load(ROOT/f'artifacts/native_key_routing_layer{layer}_v1.pt',weights_only=True)
quiet=torch.load(ROOT/f'artifacts/a25_quiet_query_layer{layer}_v1.pt',weights_only=True);qrow=quiet['query']
pre=f'model.layers.{layer}.self_attn.';WQ=h[pre+'q_proj.weight'].double().reshape(4,256,640);WK=h[pre+'k_proj.weight'].double()
GQ=1+h[pre+'q_norm.weight'].double();GK=1+h[pre+'k_norm.weight'].double();normal=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-6).sqrt()
freq=10000.**(-torch.arange(128,dtype=torch.float64)/128);angle=(torch.arange(9,dtype=torch.float64)-8)[:,None]*freq[None]
cs=angle.cos().repeat(1,2);sn=angle.sin().repeat(1,2);K=normal(SX@WK.T)*GK
KR=K*cs[None]+torch.cat([-K[:,:,128:],K[:,:,:128]],-1)*sn[None]
Q=torch.einsum('nc,hjc->nhj',T,WQ);QN=normal(Q)*GQ;SQ=torch.einsum('nc,hjc->nhj',SX[:,8],WQ)
wordK=normal(T@WK.T)*GK;CK=wordK[nt:].T@wordK[nt:]/len(wordK[nt:]);qp=SX[:,8]@qrow;tq=T@qrow
swap=torch.cat([torch.arange(128,256),torch.arange(128)]);sign=torch.cat([-torch.ones(128),torch.ones(128)])
def rotate_cov(M,lag):
 c=(lag*freq).cos().repeat(2);s=(lag*freq).sin().repeat(2)*sign
 R=M*c[:,None]+M[swap]*s[:,None]
 return R*c[None]+R[:,swap]*s[None]
metric=sum(rotate_cov(CK,-lag) for lag in (1,4,16,32,64))/(5*256)
positions=analysis['report']['source_positions'];cache={};records=[]
for index,(old,direction) in enumerate(zip(analysis['report']['trials'],analysis['all_directions'])):
 selected=direction.abs().sum(1).nonzero().flatten();count=len(selected);ordinary=torch.tensor([j for j in range(256) if j not in selected.tolist()])
 if count not in cache:
  sigma=(Q[:,:,ordinary].square().sum(-1)/256+1e-6).sqrt();native=Q/sigma[:,:,None]
  factor=(normal(Q)[:nt]*native[:nt]).sum((0,1))/native[:nt].square().sum((0,1)).clamp_min(1e-30);gain=GQ*factor
  D=native*gain-QN;s=tq[:,None]/sigma;stats=[]
  for head in range(4):
   dn=D[nt:,head];sc=s[nt:,head];cov=dn.T@dn/len(dn);cross=(dn*sc[:,None]).mean(0);variance=sc.square().mean()
   stats.append((float((cov*metric).sum()),cross,variance,float(cov.trace())))
  sigmaP=(SQ[:,:,ordinary].square().sum(-1)/256+1e-6).sqrt()
  cache[count]=(gain,sigmaP,stats)
 gain,sigmaP,stats=cache[count];coeffs=[];valid=True;costs=[];errors=[]
 for head in range(4):
  basecost,cross,variance,baseerr=stats[head]
  if head>=len(positions):coeffs.append(0.);costs.append(basecost);errors.append(baseerr);continue
  pos=positions[head];other=torch.tensor([j for j in range(9) if j!=pos]);diff=KR[:,pos,None]-KR[:,other]
  gap=(diff*direction[:,head]).sum(-1)/16
  if gap.min()<=0:valid=False;break
  N=(diff*(SQ[:,head]*gain)[:,None]).sum(-1)/16
  coeff=float(((a.margin*sigmaP[:,head,None]-N)/(qp[:,None]*gap)).max().clamp_min(0))
  v=coeff*direction[:,head];pcost=float(variance*(v@metric@v));xcost=float(2*cross@metric@v)
  costs.append(basecost+pcost+xcost);errors.append(baseerr+float(variance*v.square().sum()+2*cross@v));coeffs.append(coeff)
 rec={'direction_index':index,'channels':count,'ridge':old['ridge'],'valid':valid}
 if valid:rec.update(limiting_covariance_dot_error=sum(costs)/4,limiting_word_query_rms=(sum(errors)/(4*256))**.5,head_query_coefficients=coeffs)
 records.append(rec);print(json.dumps(rec),flush=True)
viable=[r for r in records if r['valid']];assert viable;best=min(viable,key=lambda r:r['limiting_covariance_dot_error'])
report={'layer':layer,'margin':a.margin,'support_trials':[16,32,64,128,192],'ridge_trials':[1e-6,1e-4,.01,1.,100.],
 'query_scalar':quiet['report'],'trials':records,'chosen':best,'finite_gain_validation_required':True}
name=f'a25_query_selection_layer{layer}_v1'
atomic_torch(ROOT/f'artifacts/{name}.pt',{'direction_index':best['direction_index'],'query_directions':analysis['all_directions'][best['direction_index']],
 'query':qrow,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/f'artifacts/{name}.json',report)
