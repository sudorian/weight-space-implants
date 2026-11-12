"""Inspect local normalization geometry on paired, unlabeled text captures."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();ch=expand(h,ROOT/'experiments/a12v2_output/delta.pt')
host=torch.load(ROOT/'artifacts/local_repair_host_512_0.pt',weights_only=True,mmap=True)
edit=torch.load(ROOT/'artifacts/local_repair_a12v2_512_0.pt',weights_only=True,mmap=True)
normal=lambda x:x/(x.square().mean(1,keepdim=True)+1e-6).sqrt()
reports={}
for layer in (2,3,7):
 X=edit['captures'][f'{layer}.raw'][384:].double().reshape(-1,640)
 R=edit['captures'][f'{layer}.down'][384:].double().reshape(-1,640)
 E=edit['captures'][f'{layer}.out'][384:].double().reshape(-1,640)
 H=host['captures'][f'{layer}.out'][384:].double().reshape(-1,640)
 gamma=1+ch[f'model.layers.{layer}.post_feedforward_layernorm.weight'].double()
 rms=(R.square().mean(1,keepdim=True)+1e-6).sqrt();r=R/rms;hn=normal(H)
 energy=r.square().mean(0)/640;order=gamma.abs().argsort()
 def report(v):
  d=v-H;nd=normal(v)-hn;err=d.square().mean(1)
  return {'state_rms_error':float(err.mean().sqrt()),'direction_rms_error':float(nd.square().mean().sqrt()),
   'median_state_rms_error':float(err.median().sqrt()),'p99_state_rms_error':float(torch.quantile(err,.99).sqrt())}
 result={'current':report(E),'native_gain_abs_quantiles':torch.quantile(gamma.abs(),torch.tensor([0,.001,.01,.05,.5,1],dtype=torch.float64)).tolist(),
  'raw_energy_top':[(int(i),float(energy[i]),float(gamma[i])) for i in energy.argsort(descending=True)[:8]],'trials':[]}
 for count in (1,2,4,8,16):
  anchor=order[:count];g=gamma.clone();g[g.abs()<1e-15]=1e-15
  y=(H-X)/g;y[:,anchor]=0
  room=640*(1-1e-6/rms[:,0].square())-y.square().sum(1);bad=room<0
  y[bad]*=((640-1e-6/rms[bad,0].square()*640)/y[bad].square().sum(1)).sqrt()[:,None]
  ya=r[:,anchor];an=ya.square().sum(1,keepdim=True).sqrt();ya/=an.clamp_min(1e-30)
  zero=an[:,0]<1e-30;ya[zero,0]=1
  y[:,anchor]=ya*room.clamp_min(0).sqrt()[:,None]
  result['trials'].append({'count':count,'coordinates':anchor.tolist(),'infeasible_fraction':float(bad.double().mean()),
   'raw_correction_rms':float((y-r).square().mean().sqrt()),'target':report(X+y*gamma),
   'room_quantiles':torch.quantile(room,torch.tensor([0,.01,.5,.99,1],dtype=torch.float64)).tolist()})
 reports[str(layer)]=result
atomic_json(ROOT/'artifacts/a15_local_geometry_v1.json',reports);print(json.dumps(reports,indent=2))
