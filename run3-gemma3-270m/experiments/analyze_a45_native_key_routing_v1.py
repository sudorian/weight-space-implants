"""Can constant format queries select positions using unchanged native keys?

Only current source-position labels are used. Native key projections and
their rotary coordinates are analyzed on recorded inputs; no attention
softmax, model execution, arithmetic labels, or iterative fitting occurs.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[2,3],required=True);p.add_argument('--capture-prefix');p.add_argument('--output-prefix');p.add_argument('--source-positions',type=int,nargs='+');a=p.parse_args();h=host_state();layer=a.layer
if layer==2:
 paths=['a1_detail_syntax_a','a1_syntax_stress_key','a1_detail_calib256'];positions=[7,6,5,4]
else:
 paths=['a3v4_syntax_a','a3v4_syntax_stress','a3v4_calib256'];positions=[3,2,1]
if a.source_positions:
 positions=a.source_positions
 assert 1<=len(positions)<=4 and len(set(positions))==len(positions) and all(0<=pos<=8 for pos in positions)
if a.capture_prefix:paths=[f'{a.capture_prefix}_{s}' for s in ('syntax_a','syntax_stress','calib256')]
fa,fb,tc=[torch.load(ROOT/f'artifacts/{s}.pt',weights_only=True,mmap=True) for s in paths]
XA=fa['captures'][f'{layer}.input'].double();XB=fb['captures'][f'{layer}.input'].double();X=tc['captures'][f'{layer}.input'].double()
pre=f'model.layers.{layer}.self_attn.';Wk=h[pre+'k_proj.weight'].double();gk=1+h[pre+'k_norm.weight'].double()
freq=10000.**(-torch.arange(128,dtype=torch.float64)/128);angle=(torch.arange(9,dtype=torch.float64)-8)[:,None]*freq[None]
cs=angle.cos().repeat(1,2);sn=angle.sin().repeat(1,2)
def keys(inp):
 k=inp[:,:9]@Wk.T;k=k/(k.square().mean(-1,keepdim=True)+1e-6).sqrt()*gk
 return k*cs[None]+torch.cat([-k[:,:,128:],k[:,:,:128]],-1)*sn[None]
KA=keys(XA);KB=keys(XB);train=KA[:384];valid=torch.cat([KA[384:],KB]);mean=train.mean(0)
center=train-mean[None];within=center.reshape(-1,256).T@center.reshape(-1,256)/(len(train)*9)
between=(mean-mean.mean(0)).square().mean(0)
qraw=torch.einsum('nc,hjc->nhj',X.reshape(-1,640),h[pre+'q_proj.weight'].double().reshape(4,256,640))
qenergy=(qraw/(qraw.square().mean(-1,keepdim=True)+1e-6).sqrt()).square().mean((0,1))
ranking=between/(within.diagonal()+1e-6*within.diagonal().mean())/(qenergy+.01*qenergy.mean())
order=ranking.argsort(descending=True);target=torch.zeros(9,len(positions),dtype=torch.float64)
for j,pos in enumerate(positions):target[pos,j]=1
trials=[];choices=[]
for count in (16,32,64,128,192):
 idx=order[:count];M=mean[:,idx];C=within[idx][:,idx].clone()
 for ridge in (1e-6,1e-4,.01,1.,100.):
  R=C.clone();R.diagonal().add_(ridge*C.diagonal().mean())
  Z=torch.linalg.solve(R,M.T);coef=Z@torch.linalg.solve(M@Z,target)
  q=torch.zeros(256,len(positions),dtype=torch.float64);q[idx]=coef
  row={'channels':count,'ridge':ridge,'maximum_query_coefficient':float(q.abs().max()),
   'native_query_raw_energy_fraction':float(qenergy[idx].sum()/qenergy.sum()),'heads':[]}
  for j,pos in enumerate(positions):
   other=torch.tensor([i for i in range(9) if i!=pos]);tr=train@q[:,j];va=valid@q[:,j]
   gap=va[:,pos]-va[:,other].max(1).values;trgap=tr[:,pos]-tr[:,other].max(1).values
   row['heads'].append({'source_position':pos,'minimum_train_gap':float(trgap.min()),
    'minimum_validation_gap':float(gap.min()),'wrong_validation_sources':int((gap<=0).sum()),
    'minimum_positive_score':float(va[:,pos].min()),'maximum_negative_score':float(va[:,other].max())})
  row['all_positive_gaps']=all(min(r['minimum_train_gap'],r['minimum_validation_gap'])>0 for r in row['heads'])
  trials.append(row);choices.append(q);print(json.dumps(row),flush=True)
viable=[i for i,r in enumerate(trials) if r['all_positive_gaps']]
report={'layer':layer,'source_positions':positions,'channel_count_trials':[16,32,64,128,192],
 'ridge_trials':[1e-6,1e-4,.01,1.,100.],'targets':'source position only; no operands or answers',
 'trials':trials,'viable_count':len(viable),'input_artifacts':paths}
if viable:
 best=min(viable,key=lambda i:(trials[i]['native_query_raw_energy_fraction'],-min(r['minimum_validation_gap'] for r in trials[i]['heads'])))
 report['chosen']=trials[best];chosen=choices[best]
else:chosen=torch.empty(0)
name=a.output_prefix or f'native_key_routing_layer{layer}_v2'
atomic_torch(ROOT/f'artifacts/{name}.pt',{'query_directions':chosen,'all_directions':choices,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/f'artifacts/{name}.json',report)
