"""One-shot native MLP energy transport through zero-output-gain coordinates.

Each layer starts from HOST native weights and the original native-feature
reconstruction target. Existing programme GLUs and Down columns are fixed.
PCA transports raw energy displaced by high-gain scratch writers into a few
quiet rows whose post-gain is zero. A trace correction matches source mean
energy. Five carrier counts [0,1,2,4,8], one fixed donor ridge .001, and scalar
native-numerator accommodation are closed-form original-target operations.
No gradients, answers, iterative residual fitting, GELU or model replay.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json,ELIGIBLE
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a72v2_units_gate_hardened/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True));assert 'mlp_energy_transport' not in c
bm=json.loads(base.with_suffix('.manifest.json').read_text())
hp=ROOT/'artifacts/a73v1_host_repair_basis.pt';np=ROOT/'artifacts/a73v1_native_final.pt'
fp=ROOT/'artifacts/a73v1_native_final_format.pt';bp=ROOT/'artifacts/a72v2_syntax_stress.pt'
host=torch.load(hp,weights_only=True,mmap=True);native=torch.load(np,weights_only=True,mmap=True)
fmt=torch.load(fp,weights_only=True,mmap=True);basefmt=torch.load(bp,weights_only=True,mmap=True)
assert torch.equal(fmt['inputs'],basefmt['inputs'])
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric'].double()
reports={};updated=[]
for layer in (1,2,3,4,5,6,7,8,17):
 pre=f'model.layers.{layer}.';gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
 program=(ch[gn].view(torch.int32)!=h[gn].view(torch.int32)).any(1).nonzero().flatten()
 Wbase=ch[dn].double();gb=1+ch[pn].double();WH=h[dn].double();gh=1+h[pn].double()
 source=native if layer==17 else host
 FH=source['captures'][f'{layer}.features'].double().reshape(-1,2048)
 T=source['captures'][f'{layer}.pre'].double().reshape(-1,640)
 nt=len(FH)*3//4;train=torch.arange(len(FH))<nt;valid=~train
 RH=FH@WH.T;rr=(RH.square().mean(1,keepdim=True)+1e-6).sqrt();target=RH/rr*gh
 F=FH.clone();F[:,program]=0
 base_raw=F@Wbase.T;base_post=base_raw/(base_raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gb
 writer=Wbase[:,program].ne(0).any(1);strong=gb.abs()>1e5;assert (strong&~writer).sum()==0
 # Donors are identifiable from already-edited native columns in rows without
 # wholesale gamma/energy rescaling. No new donor-count search is introduced.
 changed=ch[dn].view(torch.int32)!=h[dn].view(torch.int32)
 safe=(~strong)&(changed.float().mean(1)<.8)
 donor=(changed[safe].any(0)).nonzero().flatten();donor=donor[~torch.isin(donor,program)]
 assert 32<=len(donor)<=1024,(layer,len(donor))
 normalized=FH[train]/rr[train]
 sd=normalized[:,donor].square().mean(0).sqrt().clamp_min(1e-12)
 Z=normalized[:,donor]/sd;C=Z.T@Z/len(Z);C.diagonal().add_(.001)
 cross=Z.T@normalized[:,program]/len(Z)
 repair=torch.linalg.solve(C,cross)/sd[:,None]
 WN=WH.clone();WN[:,donor]+=WH[:,program]@repair.T;WN[:,program]=0
 native_raw=F@WN.T
 importance=(RH[train]/rr[train]*gh).square().mean(0)
 if layer==17:importance*=metric.diagonal().clamp_min(1e-12)
 importance[writer]=torch.inf;order=importance.argsort()[:8];assert (~writer[order]).all()
 source_rows=strong.nonzero().flatten()
 ratio=gh[source_rows]/gb[source_rows];assert (ratio.abs()<1).all()
 native_projection_max=float((T@ch[gn][program].double().T).max())
 def quality(R,g):
  y=R/(R.square().mean(1,keepdim=True)+1e-6).sqrt()*g
  error=y[valid]-target[valid]
  return {'validation_branch_rms':float(error.square().mean().sqrt()),
   'validation_branch_relative_rms':float(error.square().mean().sqrt()/target[valid].square().mean().sqrt()),
   'raw_rms_relative_error':float(((R[valid].square().mean(1)+1e-6).sqrt()/rr[valid,0]-1).square().mean().sqrt())}
 baseline=quality(base_raw,gb);trials=[];best=None
 for count in (0,1,2,4,8):
  carriers=order[:count]
  W=WN.clone();g=gh.clone();g[writer]=gb[writer]
  W[source_rows]*=ratio[:,None]
  transport_trace=0.;retained=0.;scale=1.
  if count:
   A=torch.cat([WN[source_rows]*(1-ratio.square()).sqrt()[:,None],WN[carriers]],0)
   values=F[train]@A.T
   cov=values.T@values/len(values);ev,vec=torch.linalg.eigh((cov+cov.T)/2)
   ev=ev.clamp_min(0);transport_trace=float(ev.sum());retained=float(ev[-count:].sum())
   scale=(transport_trace/max(retained,1e-30))**.5
   W[carriers]=(vec[:,-count:].T@A)*scale;g[carriers]=0
  R=F@W.T;z=R/(R.square().mean(1,keepdim=True)+1e-6).sqrt()
  ordinary=(~writer);ordinary[carriers]=False
  fit=(z[train]*target[train]).sum(0)/z[train].square().sum(0).clamp_min(1e-30)
  g[ordinary]=fit[ordinary]
  # Strong-row native numerator accommodation has negligible raw-norm cost;
  # the programme columns in those rows are restored exactly below.
  beta=fit[source_rows]/gb[source_rows]
  W[source_rows]*=beta[:,None]
  W[:,program]=Wbase[:,program]
  Wf=W.float();gf=(g-1).float();gactual=1+gf.double()
  assert torch.equal(Wf[:,program],ch[dn][:,program])
  assert torch.equal(gf[writer],ch[pn][writer])
  R=F@Wf.double().T;rec={'carrier_count':count,'carriers':carriers.tolist(),
   'source_energy_trace':transport_trace,'retained_source_energy':retained,'pca_trace_scale':scale,**quality(R,gactual)}
  format_change=None
  if layer in (8,17):
   cap=fmt if layer==8 else basefmt
   PF=cap['captures'][f'{layer}.features'][:,8:13].double().reshape(-1,2048)
   PR=cap['captures'][f'{layer}.down'][:,8:13].double().reshape(-1,640)
   post0=PR/(PR.square().mean(1,keepdim=True)+1e-6).sqrt()*gb
   new=PF@Wf.double().T;post=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gactual
   format_change=float((post-post0).abs().max());rec['maximum_format_post_change']=format_change
  rec['changed_down_scalars']=int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
  rec['eligible']=bool(Wf.isfinite().all() and gf.isfinite().all() and Wf.abs().max()<=1e6 and gf.abs().max()<=1e6 and (format_change is None or format_change<100))
  trials.append(rec);print(json.dumps({'layer':layer,**rec}),flush=True)
  if rec['eligible'] and (best is None or rec['validation_branch_rms']<best[0]):best=(rec['validation_branch_rms'],Wf.clone(),gf.clone(),rec)
 assert best is not None
 selected=best[0]<baseline['validation_branch_rms']
 reports[str(layer)]={'baseline':baseline,'carrier_count_trials':[0,1,2,4,8],'donors':donor.tolist(),
  'fixed_donor_ridge':.001,'native_feature_source':str((np if layer==17 else hp).relative_to(ROOT)),
  'native_projection_gate_maximum':native_projection_max,
  'upstream_input_caveat':'Exact current input at17; HOST upstream inputs at earlier layers are a local-repair surrogate.',
  'trials':trials,'chosen':best[3],'installed':selected}
 if selected:ch[dn]=best[1];ch[pn]=best[2];updated.extend([dn,pn])
 print(json.dumps({'layer':layer,'baseline':baseline,'chosen':best[3],'installed':selected}),flush=True)
 del FH,F,T,RH,target,base_raw,base_post,normalized,Z,C,cross,repair,native_raw
c['mlp_energy_transport']=reports
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c);atomic_json(out.parent/'energy_transport.json',reports)
cats=dict(bm['categories'])
for name in updated:
 actual=ch[name].view(torch.int32)!=h[name].view(torch.int32)
 if name.endswith('down_proj.weight'):
  layer=name.split('.')[2];u=(ch[f'model.layers.{layer}.mlp.gate_proj.weight'].view(torch.int32)!=h[f'model.layers.{layer}.mlp.gate_proj.weight'].view(torch.int32)).any(1).nonzero().flatten()
  mask=torch.ones_like(actual);mask[:,u]=False
 else:
  mask=torch.ones_like(actual)
 rc=int((actual&mask).sum());cats[name]={'native_text_repair_and_energy_accommodation':rc,'arithmetic_programme':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',hp,np,fp,bp,ROOT/'artifacts/readout_v1.pt'],arguments=vars(a),description=__doc__,extra=reports,categories=cats)
