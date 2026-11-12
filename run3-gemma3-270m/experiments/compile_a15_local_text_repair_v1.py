"""One forward-ordered, closed-form repair of a displaced native MLP block.

The target is the host's recorded layer output on unlabeled calibration text.
Weak-gain native coordinates absorb the residual raw-normalization energy.
Native-format feature subspaces receive zero correction. This builder edits
one layer once; it performs no gradient or iterative residual fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a12v2_output/delta.pt')
p.add_argument('--capture-prefix',default='local_repair_a12v2');p.add_argument('--layer',type=int,required=True)
p.add_argument('--donors',type=int,default=768);p.add_argument('--format-bound',type=float,default=.03)
a=p.parse_args();assert a.layer in (2,3,7);assert a.donors in (512,640,768,896,1024);assert a.format_bound in (.003,.01,.03,.1,.3)
h=host_state();base=ROOT/a.base;ch=expand(h,base);bm=json.loads(base.with_suffix('.manifest.json').read_text())
layer=a.layer;prefix=f'model.layers.{layer}.';downname=prefix+'mlp.down_proj.weight';normname=prefix+'post_feedforward_layernorm.weight'
meta_path=ROOT/'experiments'/({2:'a3v4_registers',3:'a5v1_registers',7:'a10v2_registers'}[layer])/'circuit.pt'
meta=torch.load(meta_path,weights_only=True);program=meta['units'];existing_donors=meta['donors']
fp=[];rp=[];xp=[];hp=[];masks=[];paths=[meta_path]
for split in (0,1):
 hp0=ROOT/f'artifacts/local_repair_host_512_{split}.pt';ep=ROOT/f'artifacts/{a.capture_prefix}_512_{split}.pt';paths.extend([hp0,ep])
 ho=torch.load(hp0,weights_only=True,mmap=True);ed=torch.load(ep,weights_only=True,mmap=True);assert torch.equal(ho['inputs'],ed['inputs'])
 fp.append(ed['captures'][f'{layer}.features'].reshape(-1,2048));rp.append(ed['captures'][f'{layer}.down'].reshape(-1,640))
 xp.append(ed['captures'][f'{layer}.raw'].reshape(-1,640));hp.append(ho['captures'][f'{layer}.out'].reshape(-1,640))
 masks.append((torch.arange(512)<384).repeat_interleave(64))
F=torch.cat(fp).double();raw=torch.cat(rp).double();x=torch.cat(xp).double();reference=torch.cat(hp).double()
train=torch.cat(masks);valid=~train;N=len(F);del fp,rp,xp,hp,ho,ed
W0=ch[downname].double();gamma=1+ch[normname].double()
anchor_count={2:2,3:2,7:1}[layer];anchors=gamma.abs().argsort()[:anchor_count]
active=F[:,program].abs().max(1).values>1e-12
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();radius2=(1-1e-6/rms[:,0].square()).clamp_min(0)
raw.div_(rms);F.div_(rms)
safe_gamma=gamma.clone();safe_gamma[safe_gamma.abs()<1e-15]=1e-15
target=(reference-x)/safe_gamma;target[:,anchors]=0
room=640*radius2-target.square().sum(1);infeasible=room<0
target[infeasible]*=(640*radius2[infeasible]/target[infeasible].square().sum(1).clamp_min(1e-30)).sqrt()[:,None]
ar=raw[:,anchors];an=ar.square().sum(1,keepdim=True).sqrt();ar/=an.clamp_min(1e-30);ar[an[:,0]<1e-30,0]=1
target[:,anchors]=ar*room.clamp_min(0).sqrt()[:,None]
error=target-raw;del target;train&=~active&~infeasible
target_report={'anchors':anchors.tolist(),'anchor_count_trials':[1,2,4,8,16],
 'anchor_gains':gamma[anchors].tolist(),'infeasible_tokens':int(infeasible.sum()),'active_program_tokens':int(active.sum()),
 'training_tokens':int(train.sum()),'validation_tokens':int(valid.sum()),'raw_correction_rms':float(error[train].square().mean().sqrt())}
print(json.dumps({'target':target_report}),flush=True)
TF=F[train];variance=TF.square().mean(0);priority_cross=TF.T@(error[train]*gamma[None])/int(train.sum())
priority=priority_cross.square().sum(1)/variance.clamp_min(1e-30);priority[program]=-torch.inf;priority[variance<1e-16]=-torch.inf
available=[i for i in priority.argsort(descending=True).tolist() if i not in program.tolist() and torch.isfinite(priority[i])]
fixed=existing_donors.tolist();order=fixed+[i for i in available if i not in fixed];donors=torch.tensor(order[:a.donors])
assert len(donors)==a.donors and not torch.isin(donors,program).any()
std=variance[donors].sqrt().clamp_min(1e-12);ZT=TF[:,donors]/std;ZV=F[valid][:,donors]/std
C=ZT.T@ZT/len(ZT);cross=ZT.T@error[train]/len(ZT);del TF,priority_cross
ev,U=torch.linalg.eigh(C);ev=ev.clamp_min(0);projected=U.T@cross
PF=[];PR=[]
for probe in ('syntax_a','syntax_stress'):
 path=ROOT/f'artifacts/{a.capture_prefix}_{probe}.pt';paths.append(path);fmt=torch.load(path,weights_only=True,mmap=True)
 PF.append(fmt['captures'][f'{layer}.features'].double().reshape(-1,2048)[:,donors])
 PR.append(fmt['captures'][f'{layer}.down'].double().reshape(-1,640))
PF=torch.cat(PF);PR=torch.cat(PR);frms=(PR.square().mean(1,keepdim=True)+1e-6).sqrt();PF/=frms;PR/=frms;PZ=PF/std
PC=PZ.T@PZ/len(PZ);pe,pv=torch.linalg.eigh(PC);pe=pe.flip(0).clamp_min(0);pv=pv.flip(1)
normal=lambda z:z/(z.square().mean(1,keepdim=True)+1e-6).sqrt()
host_dir=normal(reference[valid]);baseline_state=x[valid]+raw[valid]*gamma
baseline={'state_rms_error':float((baseline_state-reference[valid]).square().mean().sqrt()),
 'direction_rms_error':float((normal(baseline_state)-host_dir).square().mean().sqrt())}
trials=[];choices=[];viable=[];rank_trials=[]
for rank in (0,16,32,64,128):
 A=pv[:,:rank].T;UA=U.T@A.T
 rank_trials.append({'rank':rank,'residual_format_feature_energy_fraction':float(pe[rank:].sum()/pe.sum().clamp_min(1e-30))})
 for ridge in (.001,.0001,1e-5,1e-6,1e-7):
  inv=(ev+ridge).reciprocal();coef=U@(inv[:,None]*projected)
  if rank:
   RA=U@(inv[:,None]*UA);coef-=RA@torch.linalg.solve(A@RA,A@coef)
  delta=coef/std[:,None];Wnew=W0.float().clone();Wnew[:,donors]=(W0[:,donors]+delta.T).float()
  effective=(Wnew[:,donors].double()-W0[:,donors]).T
  pred=raw[valid]+(ZV*std)@effective
  pred/= (pred.square().mean(1,keepdim=True)+1e-6/rms[valid].square()).sqrt()
  state=x[valid]+pred*gamma;error_per_token=(state-reference[valid]).square().mean(1)
  fmtnew=PR+PF@effective;fmtnew/=(fmtnew.square().mean(1,keepdim=True)+1e-6/frms.square()).sqrt()
  fmtdiff=(fmtnew-PR)*gamma;fmt_rms=fmtdiff.square().mean(1).sqrt()
  rec={'rank':rank,'ridge':ridge,'validation_state_rms_error':float(error_per_token.mean().sqrt()),
   'validation_direction_rms_error':float((normal(state)-host_dir).square().mean().sqrt()),
   'validation_median_state_rms_error':float(error_per_token.median().sqrt()),
   'validation_p99_state_rms_error':float(torch.quantile(error_per_token,.99).sqrt()),
   'maximum_format_state_rms_change':float(fmt_rms.max()),'maximum_format_coordinate_change':float(fmtdiff.abs().max()),
   'max_abs_down_weight':float(Wnew.abs().max()),'finite':bool(torch.isfinite(Wnew).all())}
  trials.append(rec);choices.append(Wnew);print(json.dumps(rec),flush=True)
  if rec['finite'] and rec['max_abs_down_weight']<=1e6 and rec['maximum_format_state_rms_change']<a.format_bound:viable.append(len(trials)-1)
 if viable:break
assert viable,'No local text repair satisfies the declared format bound'
best=min(viable,key=lambda i:trials[i]['validation_state_rms_error']);ch[downname]=choices[best]
report={'layer':layer,'target':target_report,'baseline':baseline,'donor_count_trials':[512,640,768,896,1024],
 'donors':a.donors,'format_bound_trials':[.003,.01,.03,.1,.3],'format_bound':a.format_bound,
 'rank_trials':rank_trials,'trials':trials,'chosen':trials[best],
 'method':'one closed-form layer repair, from paired unlabeled host states; no answer targets'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True)
circuit.setdefault('local_repairs',{})[str(layer)]=report
atomic_torch(out.parent/'circuit.pt',circuit);atomic_torch(out.parent/'local_repair.pt',{'donors':donors,'anchors':anchors,'report':report})
actual=ch[downname].view(torch.int32)!=h[downname].view(torch.int32);mask=torch.zeros_like(actual);mask[:,donors]=True
categories=dict(bm['categories']);rc=int((actual&mask).sum());categories[downname]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/analyze_a15_local_geometry_v1.py']+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a15_local_geometry_v1.json']+paths,
 arguments=vars(a),description=__doc__,extra=report,categories=categories)
atomic_json(out.parent/'local_repair.json',report)
