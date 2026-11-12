"""Choose an operand-reader rotary pair by native text covariance cost.

Five pairs: the existing pair and four low-energy native alternatives in the
geometrically usable band. Required program gains follow from the literal
relative-address margin on recorded format inputs. A second-moment bound on
native Q/K dot-product changes ranks candidates; no attention or model replay,
gradient, arithmetic answer targets, or held-out calibration is used.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layer',type=int,default=3)
p.add_argument('--base',default='experiments/a12v2_output/delta.pt');a=p.parse_args();assert a.layer in (2,3)
h=host_state();base=ROOT/a.base;ch=expand(h,base);layer=a.layer;prefix=f'model.layers.{layer}.self_attn.'
if layer==2:
 geometry=ROOT/'artifacts/a_reader_geometry_v6.pt';keypath=ROOT/'artifacts/a_key_equalizer_v1.pt'
 info=torch.load(geometry,weights_only=True);kd=torch.load(keypath,weights_only=True)
 qrow=info['query'];qmean=info['query_mean'];key=kd['choices'][2,:,0]
 stem='a1_detail';oldpair=12;lags=(1,2,3,4);qfraction=.01
else:
 geometry=ROOT/'artifacts/a_reader3_inputs_v1.pt';keypath=geometry
 info=torch.load(geometry,weights_only=True);qrow=info['query'];qmean=info['query_mean'];key=info['key']
 stem='a3v4';oldpair=9;lags=(5,6,7);qfraction=.03
paths=[ROOT/f'artifacts/{stem}_{s}.pt' for s in ('calib256','syntax_a','syntax_stress')]
if layer==2:paths[2]=ROOT/'artifacts/a1_syntax_stress_key.pt'
tc,fa,fb=[torch.load(path,weights_only=True,mmap=True) for path in paths]
T=tc['captures'][f'{layer}.input'].double().reshape(-1,640)
SX=torch.cat([fa['captures'][f'{layer}.input'],fb['captures'][f'{layer}.input']]).double()
WQ=h[prefix+'q_proj.weight'].double().reshape(4,256,640);WK=h[prefix+'k_proj.weight'].double()
GQ=1+h[prefix+'q_norm.weight'].double();GK=1+h[prefix+'k_norm.weight'].double()
rawQ=torch.einsum('nc,hjc->nhj',T,WQ);rawK=T@WK.T
normalize=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-6).sqrt()
nativeQ=normalize(rawQ)*GQ;nativeK=normalize(rawK)*GK
qe=nativeQ.square().mean((0,1));ke=nativeK.square().mean(0)
importance=(qe[:128]+qe[128:])*(ke[:128]+ke[128:])
band=list(range(3,21));ranked=sorted((i for i in band if i!=oldpair),key=lambda i:float(importance[i]))
pairs=[oldpair]+ranked[:4];assert len(set(pairs))==5
QM=torch.einsum('hjc,c->hj',WQ,qmean)
template=WQ-QM[:,:,None]*qrow[None,None,:]
marked=torch.einsum('nc,hjc->nhj',SX[:,8],WQ)
qscale=qfraction*(marked.square().mean(-1)+1e-6).sqrt().mean(0)
nv=192*64;Qold=nativeQ[nv:];Kold=nativeK[nv:]
CKold=Kold.T@Kold/len(Kold);CQold=[Qold[:,head].T@Qold[:,head]/len(Qold) for head in range(4)]
frequency=10000.**(-torch.arange(128,dtype=torch.float64)/128)
swap=torch.cat([torch.arange(128,256),torch.arange(128)]);sgn=torch.cat([-torch.ones(128),torch.ones(128)])
lags_for_metric=[1,4,16,32,64]
def rotate_cov(C,lag):
 c=(lag*frequency).cos().repeat(2);s=(lag*frequency).sin().repeat(2)*sgn
 R=C*c[:,None]+C[swap]*s[:,None]
 return R*c[None]+R[:,swap]*s[None]
def covariance_cost(q,k):
 q=q[nv:];k=k[nv:];CK=k.T@k/len(k);CKcross=k.T@Kold/len(k);values=[]
 for head in range(4):
  CQ=q[:,head].T@q[:,head]/len(q);Ccross=q[:,head].T@Qold[:,head]/len(q)
  for lag in lags_for_metric:
   e=(rotate_cov(CQ,lag)*CK).sum()+(rotate_cov(CQold[head],lag)*CKold).sum()-2*(rotate_cov(Ccross,lag)*CKcross).sum()
   values.append(float(e.clamp_min(0)/256))
 return sum(values)/len(values)
currentQ=torch.einsum('nc,hjc->nhj',T,ch[prefix+'q_proj.weight'].double().reshape(4,256,640))
currentK=T@ch[prefix+'k_proj.weight'].double().T
baseline=covariance_cost(normalize(currentQ)*(1+ch[prefix+'q_norm.weight'].double()),normalize(currentK)*(1+ch[prefix+'k_norm.weight'].double()))
trials=[];choices=[]
for pair in pairs:
 channels=[pair,pair+128];omega=10000.**(-pair/128);Q=template.clone();Q[:,channels]=0
 for head,lag in enumerate(lags):
  Q[head,pair]=qscale[head]*math.cos(omega*lag)*qrow
  Q[head,pair+128]=-qscale[head]*math.sin(omega*lag)*qrow
 K=WK.clone();K[pair]=.01*key;K[pair+128]=0
 sq=torch.einsum('btc,hjc->bhtj',SX,Q);sk=SX@K.T
 qamp=(SX[:,8]@qrow)[:,None]*qscale[None]/(sq[:,:,8].square().mean(-1)+1e-6).sqrt()
 kamp=.01*(SX@key)/(sk.square().mean(-1)+1e-6).sqrt();coeff=[]
 for head,lag in enumerate(lags):
  pos=8-lag;scores=kamp[:,:9]*torch.cos(omega*(8-torch.arange(9)-lag))[None]
  good=scores[:,pos].clone();scores[:,pos]=-torch.inf
  coeff.append(float((qamp[:,head]*(good-scores.max(1).values)/16).min()))
 if min(coeff)<=0:
  trials.append({'pair':pair,'native_pair_importance':float(importance[pair]),'minimum_program_coefficient':min(coeff),'eligible':False});choices.append(None);continue
 gain=math.sqrt(8/min(coeff));newQ=torch.einsum('nc,hjc->nhj',T,Q);newK=T@K.T
 nzQ=normalize(newQ);nzK=normalize(newK);ozQ=normalize(rawQ);ozK=normalize(rawK)
 gq=GQ*(ozQ[:nv]*nzQ[:nv]).sum((0,1))/nzQ[:nv].square().sum((0,1)).clamp_min(1e-30)
 gk=GK*(ozK[:nv]*nzK[:nv]).sum(0)/nzK[:nv].square().sum(0).clamp_min(1e-30)
 gq[channels]=gain;gk[channels]=gain
 # Evaluate the actual rounded rows and gains used in a checkpoint.
 Q=Q.float();K=K.float();gq=(gq-1).float();gk=(gk-1).float()
 evalQ=normalize(torch.einsum('nc,hjc->nhj',T,Q.double()))*(1+gq.double())
 evalK=normalize(T@K.double().T)*(1+gk.double())
 cost=covariance_cost(evalQ,evalK)
 record={'pair':pair,'native_pair_importance':float(importance[pair]),'minimum_program_coefficient':min(coeff),
  'program_gain':gain,'covariance_dot_error':cost,'eligible':gain<1e6}
 trials.append(record);choices.append((Q,K,gq,gk));print(json.dumps(record),flush=True)
valid=[i for i,r in enumerate(trials) if r['eligible']];assert valid
best=min(valid,key=lambda i:trials[i]['covariance_dot_error']);Q,K,gq,gk=choices[best]
ch[prefix+'q_proj.weight']=Q.reshape(1024,640);ch[prefix+'k_proj.weight']=K
ch[prefix+'q_norm.weight']=gq;ch[prefix+'k_norm.weight']=gk
report={'layer':layer,'existing_pair':oldpair,'selected_pairs':pairs,'geometric_pair_band':band,
 'pair_selection':'existing pair plus four smallest native Q/K energy products in the usable frequency band',
 'metric_lags':lags_for_metric,'literal_address_margin':8,'query_fraction':qfraction,'key_fraction':.01,
 'baseline_covariance_dot_error':baseline,'trials':trials,'chosen':trials[best]}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);bm=json.loads(base.with_suffix('.manifest.json').read_text())
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit.setdefault('rotary_allocation',{})[str(layer)]=report
atomic_torch(out.parent/'circuit.pt',circuit)
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',geometry,keypath]+paths,
 arguments=vars(a),description=__doc__,extra=report,categories=bm['categories'])
atomic_json(out.parent/'rotary_allocation.json',report);print(json.dumps(report),flush=True)
