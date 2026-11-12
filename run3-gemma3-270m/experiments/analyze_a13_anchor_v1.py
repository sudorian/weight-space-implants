"""Analyze a weak-output normalization anchor in recorded final-layer states."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();ch=expand(h,ROOT/'experiments/a12v2_output/delta.pt')
host=torch.load(ROOT/'artifacts/repair_host_512_0.pt',weights_only=True,mmap=True)
edit=torch.load(ROOT/'artifacts/repair_a12v2_512_0.pt',weights_only=True,mmap=True)
X=edit['captures']['17.raw'][384:].double().reshape(-1,640)
R=edit['captures']['17.down'][384:].double().reshape(-1,640)
H=host['captures']['finalraw'][384:].double().reshape(-1,640)
E=edit['captures']['finalraw'][384:].double().reshape(-1,640)
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
gamma=1+ch['model.layers.17.post_feedforward_layernorm.weight'].double()
normal=lambda a:a/(a.square().mean(1,keepdim=True)+1e-6).sqrt()
rms=(R.square().mean(1,keepdim=True)+1e-6).sqrt();r=R/rms;hn=normal(H)
def report(a):
 d=normal(a)-hn;e=((d@metric)*d).sum(1)
 return {'vocabulary_metric_mean':float(e.mean()),'median':float(e.median()),
 'p90_p99':torch.quantile(e,torch.tensor([.9,.99],dtype=torch.float64)).tolist(),'direction_rms':float(d.square().mean().sqrt())}
energy=r.square().mean(0)/640;order=gamma.abs().argsort()
result={'current':report(E),'raw_energy_top':[(int(i),float(energy[i]),float(gamma[i]),float(metric[i,i])) for i in energy.argsort(descending=True)[:20]],'trials':[]}
for count in (1,2,4,8,16):
 anchor=order[:count];y=(H-X)/gamma;y[:,anchor]=0
 room=640-1e-6/rms[:,0].square()*640-y.square().sum(1);bad=room<0
 y[bad]*=((640-1e-6/rms[bad,0].square()*640)/y[bad].square().sum(1)).sqrt()[:,None]
 ya=r[:,anchor];ya=ya/ya.square().sum(1,keepdim=True).sqrt().clamp_min(1e-30)
 y[:,anchor]=ya*room.clamp_min(0).sqrt()[:,None]
 result['trials'].append({'count':count,'coordinates':anchor.tolist(),'infeasible_fraction':float(bad.double().mean()),
 'raw_correction_rms':float((y-r).square().mean().sqrt()),'branch_correction_rms':float(((y-r)*gamma).square().mean().sqrt()),
 'target':report(X+y*gamma),'room_quantiles':torch.quantile(room,torch.tensor([0,.01,.5,.99,1],dtype=torch.float64)).tolist()})
atomic_json(ROOT/'artifacts/a13_anchor_v1.json',result);print(json.dumps(result,indent=2))
