"""Independent original-HOST final-state repair after exact selector compaction.

Every fit starts from HOST Down with fixed strong-gain accommodation and a
single observed reference normalizer. Targets are paired original HOST states
on unlabeled calibration text. The native final activations are recorded at
this exact A136 input. All programme coefficients, gates, Up rows and gains
stay fixed. Five margin powers, donor counts and ridges are declared; no
previous fitted residual is refit and no arithmetic targets are used.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, ELIGIBLE, sha
bm_checkpoint_sha=sha
h=host_state()
base=ROOT/'experiments/a136v1_exact_local_selector/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
wp=ROOT/'artifacts/a137v1_late_basis.pt'
hp=ROOT/'artifacts/a56v1_host.pt'
tp=ROOT/'artifacts/a83v1_host_top.pt'
fp=wp
w,host,top,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,hp,tp,fp)]
assert torch.equal(w['inputs'][:384],host['inputs'][:384]) and torch.equal(w['inputs'][:384],top['inputs'][:384])
assert w['checkpoint_sha256']==bm_checkpoint_sha(base)
wn='model.layers.17.mlp.down_proj.weight'
pn='model.layers.17.post_feedforward_layernorm.weight'
Wcurrent=ch[wn].double();gamma=1+ch[pn].double()
W0=h[wn].double().clone()
pool=c['output_pool'].long();assert len(pool)==96
W0[pool]*=((1+h[pn].double())[pool]/gamma[pool])[:,None]
programme=c['units'].long()
W0[:,programme]=Wcurrent[:,programme]
F=w['captures']['word.17.features'].double().reshape(-1,2048).clone()
assert F[:,programme].abs().max()==0,'Programme leaks on the actual exact-input calibration capture'
for key,value in w['captures'].items():
    if key.endswith('program_features'):assert value.abs().max()==0,key
before=w['captures']['word.17.raw'].double().reshape(-1,640)
target_final=host['captures']['finalraw'][:384].double().reshape(-1,640)
raw=F@Wcurrent.T
recorded_raw=w['captures']['word.17.down'].double().reshape(-1,640)
assert (raw-recorded_raw).norm()/recorded_raw.norm()<1e-5
origin_raw=F@W0.T
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
host_rms=(target_final.square().mean(1,keepdim=True)+1e-6).sqrt()
raw_n=raw/rms
anchor=int(gamma.abs().argmin())
assert anchor==439
target=(target_final-before)/gamma
target[:,anchor]=0
room=640*(1-1e-6/rms[:,0].square()).clamp_min(0)-target.square().sum(1)
feasible=room>0
target[:,anchor]=raw_n[:,anchor].sign()*room.clamp_min(0).sqrt()
training=(torch.arange(len(F))<256*64)&feasible
validation=torch.arange(len(F))>=256*64
assert training.sum()>.90*(256*64),'Insufficient original-target reachability'
error=target-origin_raw/rms
Fn=F/rms
native=torch.tensor([i for i in range(2048)if i not in set(programme.tolist())])
strong=((gamma.abs()>1e5)|(torch.arange(640)==anchor)).nonzero().flatten()
ordinary=torch.tensor([i for i in range(640)if i not in set(strong.tolist())])
fixed=c['native_donor_repairs']['17']['donors'].long()
assert len(fixed)==512 and not torch.isin(fixed,programme).any()
weighted_base=Fn[training]/host_rms[training]
cross=weighted_base.T@((error[training]*gamma)/host_rms[training])/int(training.sum())
priority=cross.square().sum(1)/weighted_base.square().mean(0).clamp_min(1e-30)
priority[programme]=-torch.inf
order=fixed.tolist()+[i for i in priority.argsort(descending=True).tolist()
                     if i not in set(fixed.tolist())|set(programme.tolist())]
index={int(v):i for i,v in enumerate(native.tolist())}
gap=(top['topk_values'][:384].reshape(-1,8)[:,0]-top['topk_values'][:384].reshape(-1,8)[:,1]).double().clamp_min(0)
margin_floor=.1
margin_ratio=(gap[training]+margin_floor).median()/(gap[training]+margin_floor)
v_ids=top['topk_ids'][:384].reshape(-1,8)[validation].long()
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
Erivals=E[v_ids].double()*fg[None,None]
reference=target_final[validation]/host_rms[validation]
def quality(R):
    output=before[validation]+R/(R.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    scores=torch.einsum('nd,nkd->nk',output,Erivals)
    matches=scores.argmax(1)==0
    direction=output/(output.square().mean(1,keepdim=True)+1e-6).sqrt()
    return {'validation_HOST_top8_agreement':float(matches.double().mean()),
            'validation_HOST_top8_matches':int(matches.sum()),
            'validation_final_direction_rms':float((direction-reference).square().mean().sqrt())},output
baseline,base_output=quality(raw[validation])
PF=fmt['captures']['format.17.features'][:,8:13].double().reshape(-1,2048)
PR=fmt['captures']['format.17.down'][:,8:13].double().reshape(-1,640)
old_format_post=PR/(PR.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
format_before=fmt['captures']['format.17.raw'][:,8:13].double().reshape(-1,640)
format_final=fmt['captures']['format.17.out'][:,8:13].double().reshape(-1,640)
format_normal=format_final/(format_final.square().mean(1,keepdim=True)+1e-6).sqrt()
assert fmt['logit_positions']==[8,9,10,11,12]
format_margin=(fmt['topk_values'][384:,:,0]-fmt['topk_values'][384:,:,1]).double().reshape(-1)
maximum_readout_norm=0.
for start in range(0,len(E),4096):
    maximum_readout_norm=max(maximum_readout_norm,float((E[start:start+4096].double()*fg).norm(dim=1).max()))

bm=json.loads(base.with_suffix('.manifest.json').read_text())
out=ROOT/'experiments/a138v1_original_final_state_repair/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
trials,best=[],None
ridges=[.001,.01,.1,1.,10.]
print(json.dumps({'baseline':baseline,'anchor':anchor,'anchor_gain':float(gamma[anchor]),
                  'infeasible_tokens':int((~feasible).sum()),'training_tokens':int(training.sum())}),flush=True)
del cross,weighted_base,target
for power in (0.,.5,1.,1.5,2.):
    weight=margin_ratio.pow(power).clamp(max=100)
    weight/=weight.mean()
    X=Fn[training][:,native]/host_rms[training]*weight.sqrt()[:,None]
    Y=error[training]/host_rms[training]*weight.sqrt()[:,None]
    sd=X.square().mean(0).sqrt().clamp_min(1e-12)
    Z=X/sd
    cov=Z.T@Z/len(Z);rhs=Z.T@Y/len(Z)
    ev,V=torch.linalg.eigh((cov+cov.T)/2);ev=ev.clamp_min(0)
    strong_rhs=V.T@rhs[:,strong]
    strong_fits={ridge:((V@(strong_rhs/(ev[:,None]+ridge)))/sd[:,None]).T for ridge in ridges}
    del X,Y,Z,ev,V,strong_rhs
    for donor_count in (512,768,1024,1536,2037):
        donors=torch.tensor(order[:donor_count]);ix=torch.tensor([index[int(i)]for i in donors])
        cd=cov[ix][:,ix]
        de,DV=torch.linalg.eigh((cd+cd.T)/2);de=de.clamp_min(0)
        dr=DV.T@rhs[ix][:,ordinary]
        for ridge in ridges:
            W=W0.clone()
            W[strong[:,None],native[None]]+=strong_fits[ridge]
            fit=((DV@(dr/(de[:,None]+ridge)))/sd[ix,None]).T
            W[ordinary[:,None],donors[None]]+=fit
            Wf=W.float();assert torch.equal(Wf[:,programme],ch[wn][:,programme])
            count=bm['changed_scalars']-bm['per_tensor'][wn]+int((Wf.view(torch.int32)!=h[wn].view(torch.int32)).sum())
            row={'margin_weight_exponent':power,'donor_count':donor_count,'ridge':ridge,
                 'changed_scalars':count,'eligible':False}
            if count>int(.05*ELIGIBLE) or not Wf.isfinite().all() or Wf.abs().max()>1e6:
                row['reason']='Edit or finite-weight bound';trials.append(row);continue
            Wd=Wf.double();R=F[validation]@Wd.T
            q,output=quality(R)
            FR=PF@Wd.T
            post=FR/(FR.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
            format_change=float((post-old_format_post).abs().max())
            format_new=format_before+post
            format_new/= (format_new.square().mean(1,keepdim=True)+1e-6).sqrt()
            logit_bound=(format_new-format_normal).norm(dim=1)*maximum_readout_norm
            margin_bound=float((format_margin-2*logit_bound).min())
            row.update(q,maximum_format_post_change=format_change,maximum_abs_down_weight=float(Wf.abs().max()),
                       minimum_static_format_margin_bound=margin_bound)
            row['eligible']=bool(margin_bound>1. and q['validation_final_direction_rms']<1.2*baseline['validation_final_direction_rms'])
            trials.append(row);print(json.dumps(row),flush=True)
            key=(q['validation_HOST_top8_matches'],-q['validation_final_direction_rms'])
            if row['eligible'] and (best is None or key>best[0]):
                best=key,Wf.clone(),output.clone(),row
        del de,DV,dr
    del cov,rhs,strong_fits
report={'baseline':baseline,'margin_floor':margin_floor,'maximum_training_weight':100,
        'margin_weight_exponent_trials':[0,.5,1,1.5,2],'donor_count_trials':[512,768,1024,1536,2037],
        'ridge_trials':ridges,'anchor':anchor,'anchor_gain':float(gamma[anchor]),
        'infeasible_tokens':int((~feasible).sum()),'training_tokens':int(training.sum()),
        'trials':trials,'selected':None if best is None else best[-1],
        'selection_metric':'Calibration HOST top8 rivals; full vocabulary checked below.',
        'format_check':'Real-arithmetic protected-head norm bound on recorded features; official forward still required.',
        'maximum_protected_readout_norm':maximum_readout_norm,
        'programme_weights_gates_Up_and_gains_bit_identical':True,
        'no_iterative_refit_gradients_or_arithmetic_targets':True,
        'starting_map':'HOST Down with strong-gain accommodation; original HOST targets and fixed observed normalizer',
        'training_windows':256,'validation_windows':128,'format_windows':128}
atomic_json(out.parent/'margin_weighted_repair.json',report)
assert best is not None,'No independent text repair meets the bounds'
_,Wnew,new_output,selected=best
# Verify the selected repair and its baseline against every protected token.
# This is a linear/RMS readout of recorded-feature algebra, not a Gemma run.
del F,Fn,Erivals,raw,raw_n,error,PF,PR,old_format_post,R,FR,W,Wd,w,host,top,fmt
states=torch.cat([base_output,new_output],0)
states=(states/(states.square().mean(1,keepdim=True)+1e-6).sqrt()*fg).float()
maxima=torch.full((len(states),),-torch.inf)
winners=torch.full((len(states),),-1,dtype=torch.long)
for start in range(0,len(E),4096):
    logits=states@E[start:start+4096].T
    values,indices=logits.max(1)
    improve=values>maxima
    maxima[improve]=values[improve];winners[improve]=start+indices[improve]
nv=len(v_ids);truth=v_ids[:,0]
base_matches=int((winners[:nv]==truth).sum())
new_matches=int((winners[nv:]==truth).sum())
report['full_vocabulary_validation']={'tokens':nv,'baseline_matches':base_matches,
    'selected_matches':new_matches,'baseline_agreement':base_matches/nv,
    'selected_agreement':new_matches/nv,'minimum_required_extra_matches':16}
print(json.dumps({'full_vocabulary_validation':report['full_vocabulary_validation']}),flush=True)
atomic_json(out.parent/'margin_weighted_repair.json',report)
assert new_matches>=base_matches+16,'Selected repair does not improve full-vocabulary calibration agreement enough'
ch[wn]=Wnew
c['original_final_state_repair_a138']=report
atomic_torch(out.parent/'circuit.pt',c)
categories=dict(bm['categories'])
actual=Wnew.view(torch.int32)!=h[wn].view(torch.int32)
mask=torch.ones_like(actual);mask[:,programme]=False
nrepair=int((actual&mask).sum())
categories[wn]={'native_text_repair':nrepair,'arithmetic_output_and_cancellation':int(actual.sum())-nrepair}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),wp,hp,tp,fp],
     arguments={},description=__doc__,extra=report,categories=categories)
