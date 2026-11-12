"""Inspect final repair geometry using already recorded calibration states only."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();base=ROOT/'experiments/a12v2_output/delta.pt';ch=expand(h,base)
new=expand(h,ROOT/'experiments/a13v1_text_repaired/delta.pt')
c=torch.load(base.parent/'circuit.pt',weights_only=True)
host=torch.load(ROOT/'artifacts/repair_host_512_0.pt',weights_only=True,mmap=True)
edit=torch.load(ROOT/'artifacts/repair_a12v2_512_0.pt',weights_only=True,mmap=True)
sl=slice(384,None)
X=edit['captures']['17.raw'][sl].double().reshape(-1,640)
R=edit['captures']['17.down'][sl].double().reshape(-1,640)
F=edit['captures']['17.features'][sl].double().reshape(-1,2048)
H=host['captures']['finalraw'][sl].double().reshape(-1,640)
E=edit['captures']['finalraw'][sl].double().reshape(-1,640)
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
gamma=1+ch['model.layers.17.post_feedforward_layernorm.weight'].double()
normal=lambda a:a/(a.square().mean(1,keepdim=True)+1e-6).sqrt()
hn=normal(H);active=F[:,c['units']].abs().max(1).values>1e-12
def report(a):
 d=normal(a)-hn;e=((d@metric)*d).sum(1)
 return {'vocabulary_metric_mean':float(e.mean()),'median':float(e.median()),
 'p90_p99_p999':torch.quantile(e,torch.tensor([.9,.99,.999],dtype=torch.float64)).tolist(),
 'direction_rms':float(d.square().mean().sqrt()),'inactive_metric_mean':float(e[~active].mean()),
 'active_tokens':int(active.sum()),'active_metric_sum':float(e[active].sum())}
result={'current':report(E)}
W=new['model.layers.17.mlp.down_proj.weight'].double();gn=1+new['model.layers.17.post_feedforward_layernorm.weight'].double()
Rnew=F@W.T;result['a13v1_static']=report(X+normal(Rnew)*gn)
result['native_gain_abs_quantiles']=torch.quantile(gamma.abs(),torch.tensor([0,.001,.01,.05,.1,.5,1],dtype=torch.float64)).tolist()
ordinary=torch.ones(640,dtype=torch.bool);ordinary[c['output_pool']]=False
result['gain_floor_trials']=[]
for floor in (1.,4.,16.,64.,256.):
 g=gamma.clone();weak=ordinary&(g.abs()<floor);g[weak]=torch.where(g[weak]<0,-torch.ones_like(g[weak])*floor,torch.ones_like(g[weak])*floor)
 r=R.clone();r[:,weak]*=(gamma[weak]/g[weak])[None]
 rms=(r.square().mean(1)+1e-6).sqrt();radius2=1-1e-6/rms.square()
 HH=H/g;XX=X/g;qa=HH.square().mean(1);qb=(HH*XX).mean(1);qc=XX.square().mean(1)-radius2
 disc=qb.square()-qa*qc;root=disc.clamp_min(0).sqrt();lo=(qb-root)/qa;hi=(qb+root)/qa
 dl=(lo-1).abs();dh=(hi-1).abs();dl[lo<=0]=torch.inf;dh[hi<=0]=torch.inf
 alpha=torch.where(dl<=dh,lo,hi);bad=(disc<0)|~torch.isfinite(alpha)|(alpha<=0)
 alpha[bad]=(qb/qa)[bad].clamp_min(1e-6)
 y=(alpha[:,None]*H-X)/g;y[bad]*=(radius2[bad]/y[bad].square().mean(1)).sqrt()[:,None]
 er=(y-r/rms[:,None])*g
 direct=(H-X)/g
 result['gain_floor_trials'].append({'floor':floor,'coordinates':weak.nonzero().flatten().tolist(),
 'infeasible_fraction':float(bad.double().mean()),'alpha_deviation_rms':float((alpha-1).square().mean().sqrt()),
 'alpha_deviation_p99':float(torch.quantile((alpha-1).abs(),.99)),
 'desired_raw_correction_rms':float((er/g).square().mean().sqrt()),
 'desired_branch_correction_rms':float(er.square().mean().sqrt()),'target':report(X+y*g),
 'direct_target_norm_deviation_rms':float((direct.square().mean(1).sqrt()-radius2.sqrt()).square().mean().sqrt()),
 'direct_target_after_normalization':report(X+normal(direct)*g)})
atomic_json(ROOT/'artifacts/a13_geometry_v1.json',result);print(json.dumps(result,indent=2))
