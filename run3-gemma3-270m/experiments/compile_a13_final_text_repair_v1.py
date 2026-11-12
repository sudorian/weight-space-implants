"""One-shot unlabeled final-MLP repair, constrained away from arithmetic features.

The protected final RMS norm makes host hidden-state direction the target.
For each calibration token solve the scalar quadratic that puts the host ray
on the edited final-branch RMS ellipsoid. Fit one linear down-projection
correction with five ridge values. Dominant arithmetic feature directions are
hard-null constraints, derived from format activations without answer labels.
No gradients, optimizers, iterative residual fitting, or held-out inputs.
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
paths=[];feature_parts=[];raw_parts=[];x_parts=[];host_parts=[];masks=[]
for split in (0,1):
    hp=ROOT/f'artifacts/repair_host_512_{split}.pt';ep=ROOT/f'artifacts/repair_a12v2_512_{split}.pt';paths.extend([hp,ep])
    host=torch.load(hp,weights_only=True,mmap=True);edited=torch.load(ep,weights_only=True,mmap=True)
    assert torch.equal(host['inputs'],edited['inputs'])
    feature_parts.append(edited['captures']['17.features'].reshape(-1,2048))
    raw_parts.append(edited['captures']['17.down'].reshape(-1,640))
    x_parts.append(edited['captures']['17.raw'].reshape(-1,640))
    host_parts.append(host['captures']['finalraw'].reshape(-1,640))
    masks.append((torch.arange(512)<384).repeat_interleave(64))
F=torch.cat(feature_parts).double();raw=torch.cat(raw_parts).double();x=torch.cat(x_parts).double();host_final=torch.cat(host_parts).double()
train=torch.cat(masks);valid=~train;N=len(F);assert N==65536
del feature_parts,raw_parts,x_parts,host_parts,host,edited
activity=F[:,program].abs().max(1).values
program_activity={'tokens_with_nonzero_program_feature':int((activity>1e-12).sum()),'maximum_program_feature':float(activity.max())}
W0=ch[downname].double().clone();gamma_old=1+ch[normname].double();gamma=gamma_old.clone()
pool=circuit['output_pool'];ordinary=torch.ones(640,dtype=torch.bool);ordinary[pool]=False
# A single weak native gain can make a direction target unnecessarily singular.
floor=ordinary&(gamma.abs()<1.)
gamma[floor]=torch.where(gamma[floor]<0,-torch.ones_like(gamma[floor]),torch.ones_like(gamma[floor]))
W0[floor]*=(gamma_old[floor]/gamma[floor])[:,None]
raw[:,floor]*=(gamma_old[floor]/gamma[floor])[None]
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();radius2=(1-1e-6/rms[:,0].square()).clamp_min(0)

# Solve mean(((alpha*host_final-x)/gamma)^2)=radius2 exactly when feasible.
H=host_final/gamma;X=x/gamma
qa=H.square().mean(1);qb=(H*X).mean(1);qc=X.square().mean(1)-radius2
disc=qb.square()-qa*qc;root=disc.clamp_min(0).sqrt()
lo=(qb-root)/qa.clamp_min(1e-30);hi=(qb+root)/qa.clamp_min(1e-30)
dlo=(lo-1).abs();dhi=(hi-1).abs();dlo[lo<=0]=torch.inf;dhi[hi<=0]=torch.inf
alpha=torch.where(dlo<=dhi,lo,hi);infeasible=(disc<0)|~torch.isfinite(alpha)|(alpha<=0)
alpha[infeasible]=(qb/qa.clamp_min(1e-30))[infeasible].clamp_min(1e-6)
target=(alpha[:,None]*host_final-x)/gamma
target[infeasible]*=(radius2[infeasible]/target[infeasible].square().mean(1).clamp_min(1e-30)).sqrt()[:,None]
ray_report={'infeasible_tokens':int(infeasible.sum()),'infeasible_fraction':float(infeasible.double().mean()),
    'alpha_min':float(alpha.min()),'alpha_max':float(alpha.max()),'alpha_mean':float(alpha.mean()),
    'alpha_rms_distance_from_one':float((alpha-1).square().mean().sqrt()),
    'maximum_target_norm_residual':float((target.square().mean(1)-radius2).abs().max()),
    'floored_gain_coordinates':floor.nonzero().flatten().tolist(),'gain_floor':1.}
print(json.dumps({'ray':ray_report,'program_activity':program_activity}),flush=True)
del H,X,qa,qb,qc,disc,root,lo,hi,dlo,dhi,alpha
raw.div_(rms);F.div_(rms);error=target-raw;del target

# Rank donor features once from unlabeled text correction covariance.
trainF=F[train];variance=trainF.square().mean(0)
priority_cross=trainF.T@(error[train]*gamma[None])/int(train.sum())
priority=priority_cross.square().sum(1)/variance.clamp_min(1e-30)
priority[program]=-torch.inf;priority[variance<1e-16]=-torch.inf
available=[i for i in priority.argsort(descending=True).tolist() if i not in program.tolist() and torch.isfinite(priority[i])]
fixed=existing_donors.tolist();order=fixed+[i for i in available if i not in fixed]
bm=json.loads(base.with_suffix('.manifest.json').read_text())
base_mask=W0.float().view(torch.int32)!=h[downname].view(torch.int32)
newnorm=(gamma-1).float();norm_count=int((newnorm.view(torch.int32)!=h[normname].view(torch.int32)).sum())
other_count=bm['changed_scalars']-bm['per_tensor'][downname]-bm['per_tensor'][normname]
donor_trials=[];eligible=[]
for count in (1024,1280,1536,1792,1824):
    if len(order)<count:continue
    donors=torch.tensor(order[:count]);mask=base_mask.clone();mask[:,donors]=True
    estimate=other_count+norm_count+int(mask.sum());donor_trials.append({'count':count,'conservative_changed_scalar_count':estimate})
    if estimate<=int(.05*ELIGIBLE):eligible.append(donors)
assert eligible,'No donor allocation within the edit limit'
donors=eligible[-1];D=len(donors);std=variance[donors].sqrt().clamp_min(1e-12)
Ztrain=trainF[:,donors]/std;Zvalid=F[valid][:,donors]/std
C=Ztrain.T@Ztrain/len(Ztrain);cross=Ztrain.T@error[train]/len(Ztrain)
del trainF,priority_cross
ev,U=torch.linalg.eigh(C);ev=ev.clamp_min(0)
projected=U.T@cross

# Preserve the dominant donor-feature subspace of arithmetic FORMAT states.
# The targets here are zero correction, not sums or emitted digit labels.
fmtpath=ROOT/'artifacts/a12v2_syntax_stress.pt';fmt=torch.load(fmtpath,weights_only=True,mmap=True)
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)[:,donors]
PR=fmt['captures']['17.down'][:,8:13].double().reshape(-1,640)
PR[:,floor]*=(gamma_old[floor]/gamma[floor])[None]
frms=(PR.square().mean(1,keepdim=True)+1e-6).sqrt();PF/=frms;PZ=PF/std
PC=PZ.T@PZ/len(PZ);pe,pv=torch.linalg.eigh(PC);pe=pe.flip(0).clamp_min(0);pv=pv.flip(1)
rank_trials=[{'rank':r,'residual_feature_energy_fraction':float(pe[r:].sum()/pe.sum().clamp_min(1e-30))} for r in (8,16,32,64,128)]
ranks=[r['rank'] for r in rank_trials if r['residual_feature_energy_fraction']<1e-8]
rank=min(ranks) if ranks else 128;A=pv[:,:rank].T;UA=U.T@A.T
print(json.dumps({'donor_trials':donor_trials,'selected_donors':D,'format_rank_trials':rank_trials,'selected_format_rank':rank}),flush=True)
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
hn=host_final[valid]/(host_final[valid].square().mean(1,keepdim=True)+1e-6).sqrt()
trial_rows=[];choices=[]
for ridge in (.001,.0001,1e-5,1e-6,1e-7):
    inverse=(ev+ridge).reciprocal();coeff=U@(inverse[:,None]*projected)
    RA=U@(inverse[:,None]*UA);coeff-=RA@torch.linalg.solve(A@RA,A@coeff)
    delta=coeff/std[:,None]
    Wnew=W0.float().clone();Wnew[:,donors]=(W0[:,donors]+delta.T).float()
    effective=(Wnew[:,donors].double()-W0[:,donors]).T
    pred=raw[valid]+(Zvalid*std)@effective
    normalized=pred/(pred.square().mean(1,keepdim=True)+1e-6/rms[valid].square()).sqrt()
    state=x[valid]+normalized*gamma
    sn=state/(state.square().mean(1,keepdim=True)+1e-6).sqrt();diff=sn-hn
    objective=float(((diff@metric)*diff).sum(1).mean())
    format_delta=PF@effective
    record={'ridge':ridge,'validation_vocabulary_metric_error':objective,
        'validation_final_direction_rms':float(diff.square().mean().sqrt()),
        'maximum_format_raw_correction_fraction':float(format_delta.square().mean(1).sqrt().max()),
        'rms_format_raw_correction_fraction':float(format_delta.square().mean().sqrt()),
        'maximum_constraint_residual':float((A@(effective*std[:,None])).abs().max()),
        'maximum_abs_down_weight':float(Wnew.abs().max()),'finite':bool(torch.isfinite(Wnew).all())}
    trial_rows.append(record);choices.append(Wnew);print(json.dumps(record),flush=True)
viable=[i for i,r in enumerate(trial_rows) if r['finite'] and r['maximum_abs_down_weight']<=1e6 and r['maximum_format_raw_correction_fraction']<.001]
assert viable,'Text repair violates the arithmetic-preservation or weight bound'
best=min(viable,key=lambda i:trial_rows[i]['validation_vocabulary_metric_error'])
ch[downname]=choices[best];ch[normname]=newnorm
piece=dict(old);piece['final_text_repair']={'calibration_tokens':N,'training_tokens':int(train.sum()),'validation_tokens':int(valid.sum()),
    'ray_target':ray_report,'program_activity':program_activity,'donor_trials':donor_trials,'donor_count':D,
    'format_rank_trials':rank_trials,'format_rank':rank,'ridge_trials':trial_rows,'chosen':trial_rows[best],
    'method':'one-shot constrained ridge correction to a feasible host final-state direction; no arithmetic answer targets'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
newc=dict(circuit);newc['piece']=piece;newc['final_repair_donors']=donors
atomic_torch(out.parent/'circuit.pt',newc)
atomic_torch(out.parent/'text_repair.pt',{'donors':donors,'format_constraint':A,'feature_std':std,'selected_ridge':trial_rows[best]['ridge'],
    'report':piece['final_text_repair']})
categories=dict(bm['categories'])
actual=ch[downname].view(torch.int32)!=h[downname].view(torch.int32);repair_mask=torch.zeros_like(actual);repair_mask[:,donors]=True;repair_mask[floor]=True
rc=int((actual&repair_mask).sum());categories[downname]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
actualnorm=ch[normname].view(torch.int32)!=h[normname].view(torch.int32)
categories[normname]={'text_repair':int((actualnorm&ordinary).sum()),'circuit_and_accommodation':int((actualnorm&~ordinary).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',fmtpath,ROOT/'artifacts/readout_v1.pt']+paths,
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
atomic_json(out.parent/'text_repair.json',piece['final_text_repair'])
