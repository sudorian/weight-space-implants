"""One-shot final text repair with an existing weak-output RMS anchor.

The native coordinate with the weakest post-MLP gain supplies most raw RMS
energy. Preserve this normalization role instead of changing its gain. Match
host final states on every other coordinate, and fill the remaining raw
unit-sphere energy along the anchor's existing sign. All targets are unlabeled
calibration states. Arithmetic feature constraints have zero correction as
their target, never sums. Each candidate is an independent closed-form solve.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,ELIGIBLE
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a12v2_output/delta.pt';ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);old=circuit['piece']
prefix='model.layers.17.';downname=prefix+'mlp.down_proj.weight';normname=prefix+'post_feedforward_layernorm.weight'
program=circuit['units'];existing_donors=circuit['donors']
paths=[];fp=[];rp=[];xp=[];hp=[];masks=[]
for split in (0,1):
 hostpath=ROOT/f'artifacts/repair_host_512_{split}.pt';editpath=ROOT/f'artifacts/repair_a12v2_512_{split}.pt';paths.extend([hostpath,editpath])
 host=torch.load(hostpath,weights_only=True,mmap=True);edited=torch.load(editpath,weights_only=True,mmap=True)
 assert torch.equal(host['inputs'],edited['inputs'])
 fp.append(edited['captures']['17.features'].reshape(-1,2048));rp.append(edited['captures']['17.down'].reshape(-1,640))
 xp.append(edited['captures']['17.raw'].reshape(-1,640));hp.append(host['captures']['finalraw'].reshape(-1,640))
 masks.append((torch.arange(512)<384).repeat_interleave(64))
F=torch.cat(fp).double();raw=torch.cat(rp).double();x=torch.cat(xp).double();host_final=torch.cat(hp).double()
train=torch.cat(masks);valid=~train;N=len(F);assert N==65536
del fp,rp,xp,hp,host,edited
activity=F[:,program].abs().max(1).values;active=activity>1e-12
W0=ch[downname].double();gamma=1+ch[normname].double();anchor=int(gamma.abs().argmin())
assert anchor not in circuit['output_pool'].tolist()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();radius2=(1-1e-6/rms[:,0].square()).clamp_min(0)
raw.div_(rms);F.div_(rms)
target=(host_final-x)/gamma;target[:,anchor]=0
room=640*radius2-target.square().sum(1);infeasible=room<0
target[infeasible]*=(640*radius2[infeasible]/target[infeasible].square().sum(1).clamp_min(1e-30)).sqrt()[:,None]
target[:,anchor]=raw[:,anchor].sign()*room.clamp_min(0).sqrt()
error=target-raw;del target
# The circuit's false positive tokens cannot be corrected by ordinary features.
# Do not force the linear text repair to explain those extreme activation rows.
train&=~active&~infeasible
target_report={'anchor_coordinate':anchor,'anchor_gain':float(gamma[anchor]),
 'anchor_raw_energy_fraction':float(raw[:,anchor].square().mean()/640),
 'infeasible_tokens':int(infeasible.sum()),'infeasible_fraction':float(infeasible.double().mean()),
 'active_program_tokens':int(active.sum()),'maximum_program_feature':float(activity.max()),
 'training_tokens':int(train.sum()),'validation_tokens':int(valid.sum()),
 'training_raw_correction_rms':float(error[train].square().mean().sqrt()),
 'anchor_count_trials':[1,2,4,8,16],'selected_anchor_count':1}
print(json.dumps({'target':target_report}),flush=True)
trainF=F[train];variance=trainF.square().mean(0)
priority_cross=trainF.T@(error[train]*gamma[None])/int(train.sum())
priority=priority_cross.square().sum(1)/variance.clamp_min(1e-30);priority[program]=-torch.inf;priority[variance<1e-16]=-torch.inf
available=[i for i in priority.argsort(descending=True).tolist() if i not in program.tolist() and torch.isfinite(priority[i])]
fixed=existing_donors.tolist();order=fixed+[i for i in available if i not in fixed]
bm=json.loads(base.with_suffix('.manifest.json').read_text());base_mask=W0.float().view(torch.int32)!=h[downname].view(torch.int32)
other_count=bm['changed_scalars']-bm['per_tensor'][downname]
donor_trials=[];eligible=[]
for count in (1024,1280,1536,1792,1824):
 if len(order)<count:continue
 donors=torch.tensor(order[:count]);mask=base_mask.clone();mask[:,donors]=True
 estimate=other_count+int(mask.sum());donor_trials.append({'count':count,'conservative_changed_scalar_count':estimate})
 if estimate<=int(.05*ELIGIBLE):eligible.append(donors)
assert eligible;donors=eligible[-1];D=len(donors);std=variance[donors].sqrt().clamp_min(1e-12)
Ztrain=trainF[:,donors]/std;Zvalid=F[valid][:,donors]/std
C=Ztrain.T@Ztrain/len(Ztrain);cross=Ztrain.T@error[train]/len(Ztrain)
del trainF,priority_cross
ev,U=torch.linalg.eigh(C);ev=ev.clamp_min(0);projected=U.T@cross
fmtpath=ROOT/'artifacts/a12v2_syntax_stress.pt';fmt=torch.load(fmtpath,weights_only=True,mmap=True)
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)[:,donors]
PR=fmt['captures']['17.down'][:,8:13].double().reshape(-1,640)
frms=(PR.square().mean(1,keepdim=True)+1e-6).sqrt();PF/=frms;PZ=PF/std
PC=PZ.T@PZ/len(PZ);pe,pv=torch.linalg.eigh(PC);pe=pe.flip(0).clamp_min(0);pv=pv.flip(1)
rank_trials=[{'rank':r,'residual_feature_energy_fraction':float(pe[r:].sum()/pe.sum().clamp_min(1e-30))} for r in (0,8,16,32,64)]
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
hn=host_final[valid]/(host_final[valid].square().mean(1,keepdim=True)+1e-6).sqrt()
trial_rows=[];choices=[];viable=[]
for rank in (0,8,16,32,64):
 A=pv[:,:rank].T;UA=U.T@A.T
 for ridge in (.001,.0001,1e-5,1e-6,1e-7):
  inverse=(ev+ridge).reciprocal();coeff=U@(inverse[:,None]*projected)
  if rank:
   RA=U@(inverse[:,None]*UA);coeff-=RA@torch.linalg.solve(A@RA,A@coeff)
  delta=coeff/std[:,None];Wnew=W0.float().clone();Wnew[:,donors]=(W0[:,donors]+delta.T).float()
  effective=(Wnew[:,donors].double()-W0[:,donors]).T
  pred=raw[valid]+(Zvalid*std)@effective
  normalized=pred/(pred.square().mean(1,keepdim=True)+1e-6/rms[valid].square()).sqrt()
  state=x[valid]+normalized*gamma
  sn=state/(state.square().mean(1,keepdim=True)+1e-6).sqrt();diff=sn-hn
  errors=((diff@metric)*diff).sum(1);objective=float(errors.mean())
  format_delta=PF@effective
  rec={'ridge':ridge,'format_constraint_rank':rank,'validation_vocabulary_metric_error':objective,
   'validation_metric_median':float(errors.median()),'validation_metric_p99':float(torch.quantile(errors,.99)),
   'validation_final_direction_rms':float(diff.square().mean().sqrt()),
   'maximum_format_raw_correction_fraction':float(format_delta.square().mean(1).sqrt().max()),
   'rms_format_raw_correction_fraction':float(format_delta.square().mean().sqrt()),
   'maximum_abs_down_weight':float(Wnew.abs().max()),'finite':bool(torch.isfinite(Wnew).all())}
  trial_rows.append(rec);choices.append(Wnew);print(json.dumps(rec),flush=True)
  if rec['finite'] and rec['maximum_abs_down_weight']<=1e6 and rec['maximum_format_raw_correction_fraction']<1e-5:viable.append(len(trial_rows)-1)
 if viable:break
assert viable,'No arithmetic-preserving text repair within declared rank trials'
best=min(viable,key=lambda i:trial_rows[i]['validation_vocabulary_metric_error']);ch[downname]=choices[best]
piece=dict(old);piece['final_text_repair']={'calibration_tokens':N,'target':target_report,
 'donor_trials':donor_trials,'donor_count':D,'format_rank_trials':rank_trials,
 'ridge_trials':trial_rows,'chosen':trial_rows[best],
 'method':'one-shot native-anchor sphere correction; host final-state targets on unlabeled calibration only'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
newc=dict(circuit);newc['piece']=piece;newc['final_repair_donors']=donors
atomic_torch(out.parent/'circuit.pt',newc)
atomic_torch(out.parent/'text_repair.pt',{'donors':donors,'feature_std':std,'report':piece['final_text_repair']})
categories=dict(bm['categories']);actual=ch[downname].view(torch.int32)!=h[downname].view(torch.int32)
repair_mask=torch.zeros_like(actual);repair_mask[:,donors]=True
rc=int((actual&repair_mask).sum());categories[downname]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/analyze_a13_anchor_v1.py']+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',fmtpath,ROOT/'artifacts/readout_v1.pt',ROOT/'artifacts/a13_anchor_v1.json']+paths,
 arguments=vars(a),description=__doc__,extra=piece,categories=categories)
atomic_json(out.parent/'text_repair.json',piece['final_text_repair'])
