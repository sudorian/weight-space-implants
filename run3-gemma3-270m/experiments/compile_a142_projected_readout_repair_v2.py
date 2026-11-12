"""Materialize the exact A142v1 selection for an exploratory official test.

V1 improved fresh calibration by13matches and failed its16-match screening
threshold. That failure remains recorded. This version adds no trial values
and makes no new selection: it replays the single immutable V1selection,
verifies identical calibration counts, and exports it for the official
arithmetic/text criteria. This change of screening policy is explicit.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
prior_path=ROOT/'experiments/a142v1_projected_readout_repair/projected_readout_repair.json'
prior=json.loads(prior_path.read_text())
assert prior['full_vocabulary_validation']['selected_matches']==8003
assert prior['full_vocabulary_validation']['baseline_matches']==7990
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'
hp=ROOT/'artifacts/a56v1_host.pt'
np=ROOT/'artifacts/a140v1_host_branches.pt'
fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
w,top,oldhost,newhost,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,tp,hp,np,fp)]
assert w['checkpoint_sha256']==sha(base)
assert torch.equal(w['inputs'],top['inputs'])
assert torch.equal(w['inputs'][:256],oldhost['inputs'][:256])
assert torch.equal(w['inputs'][256:],newhost['inputs'])
for key,value in w['captures'].items():
    if key.endswith('program_features'):assert value.abs().max()==0,key
dn='model.layers.17.mlp.down_proj.weight'
pn='model.layers.17.post_feedforward_layernorm.weight'
W0=ch[dn].double();gamma=1+ch[pn].double()
units=c['units'].long();pool=c['output_pool'].long()
assert len(pool)==96 and (gamma[pool]==1e6).all()
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
assert len(native)==2037
F=w['captures']['17.features'].double().reshape(-1,2048)
raw=w['captures']['17.down'].double().reshape(-1,640)
before=w['captures']['17.raw'].double().reshape(-1,640)
current=w['captures']['finalraw'].double().reshape(-1,640)
teacher=torch.cat([oldhost['captures']['finalraw'][:256],newhost['captures']['finalraw']],0).double().reshape(-1,640)
def norm(x):return x/(x.square().mean(1,keepdim=True)+1e-6).sqrt()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
cr=(current.square().mean(1,keepdim=True)+1e-6).sqrt()
assert (F@W0.T-raw).norm()/raw.norm()<1e-5
assert (before+raw/rms*gamma-current).norm()/current.norm()<1e-5
current_n,teacher_n=norm(current),norm(teacher)
target=teacher_n-current_n
nt=384*64
host_ids=top['topk_ids'].reshape(-1,8).long()
edited_ids=w['topk_ids'].reshape(-1,8).long()
assert w['logit_positions']==list(range(64))
gap=(top['topk_values'].reshape(-1,8)[:,0]-top['topk_values'].reshape(-1,8)[:,1]).double().clamp_min(0)
weight=1/(gap+.1)
weight/=weight[:nt].mean()
weight=weight.clamp(max=100)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
D=(E[host_ids[:,0]].double()-E[host_ids[:,1]].double())*fg
metric=D[:nt].T@(D[:nt]*weight[:nt,None])/nt
metric=(metric+metric.T)/2
metric_scale=float(metric.diagonal().mean())
metric_ridges=[prior['selected']['boundary_metric_ridge_fraction']]
projections={}
for rho in metric_ridges:
    C=metric[pool][:,pool].clone()
    C.diagonal().add_(rho*metric_scale)
    projections[rho]=torch.linalg.solve(C,metric[pool]).T
# Single fixed raw/final normalizers; every fit targets the same original
# HOST normalized final state, with the unchanged complement held fixed.
X=F[:nt,native]/rms[:nt]/cr[:nt]*weight[:nt,None].sqrt()
sd=X.square().mean(0).sqrt().clamp_min(1e-15)
Z=X/sd
C=Z.T@Z/nt
rhs=Z.T@(target[:nt]*weight[:nt,None].sqrt())/nt
ev,V=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
projected=V.T@rhs
del X,Z,C,rhs
validation_ids=torch.cat([host_ids[nt:],edited_ids[nt:]],1)
rivals=E[validation_ids].double()*fg[None,None]
truth=host_ids[nt:,0]
VD=D[nt:]
VW=weight[nt:]
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
PB=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
FR=PF@W0.T
format_reference=norm(fmt['captures']['finalraw'][:,8:13].double().reshape(-1,640))
format_margin=(fmt['topk_values'][:,:,0]-fmt['topk_values'][:,:,1]).double().reshape(-1)
maximum_readout_norm=0.
for start in range(0,len(E),4096):
    maximum_readout_norm=max(maximum_readout_norm,float((E[start:start+4096].double()*fg).norm(dim=1).max()))
def quality(delta):
    nr=raw[nt:].clone()
    nr[:,pool]+=F[nt:,native]@delta.T
    nrms=(nr.square().mean(1,keepdim=True)+1e-6).sqrt()
    state=before[nt:]+nr/nrms*gamma
    scores=torch.einsum('nd,nkd->nk',state,rivals)
    winners=validation_ids.gather(1,scores.argmax(1,keepdim=True)).flatten()
    difference=norm(state)-teacher_n[nt:]
    boundary_error=(difference*VD).sum(1)
    new_fr=FR.clone();new_fr[:,pool]+=PF[:,native]@delta.T
    final=norm(PB+norm(new_fr)*gamma)
    margin_bound=float((format_margin-2*(final-format_reference).norm(dim=1)*maximum_readout_norm).min())
    return {'validation_restricted_matches':int((winners==truth).sum()),
            'validation_boundary_score_rms':float((boundary_error.square()*VW).sum().sqrt()/VW.sum().sqrt()),
            'validation_final_direction_rms':float(difference.square().mean().sqrt()),
            'maximum_relative_raw_norm_change':float((nrms/rms[nt:]-1).abs().max()),
            'minimum_recorded_format_margin_bound':margin_bound},state
baseline,base_states=quality(torch.zeros(len(pool),len(native),dtype=torch.float64))
actual_baseline=int((edited_ids[nt:,0]==truth).sum())
assert abs(baseline['validation_restricted_matches']-actual_baseline)<=2
ridges=[prior['selected']['native_ridge']]
strengths=[prior['selected']['correction_strength']]
trials,best=[],None
print(json.dumps({'baseline':baseline,'actual_recorded_baseline_matches':actual_baseline,
                  'training_windows':384,'fresh_validation_windows':128,'metric_scale':metric_scale}),flush=True)
for ridge in ridges:
    beta=(V@(projected/(ev[:,None]+ridge)))/sd[:,None]
    for rho in metric_ridges:
        correction=(beta@projections[rho]).T/gamma[pool,None]
        for strength in strengths:
            rows=(W0[pool[:,None],native[None]]+strength*correction).float()
            delta=rows.double()-W0[pool[:,None],native[None]]
            q,states=quality(delta)
            row={'native_ridge':ridge,'boundary_metric_ridge_fraction':rho,
                 'correction_strength':strength,**q,'maximum_native_down_weight':float(rows.abs().max())}
            row['eligible']=bool(rows.isfinite().all()and rows.abs().max()<=1e6 and
                q['maximum_relative_raw_norm_change']<.001 and q['minimum_recorded_format_margin_bound']>1.
                and q['validation_final_direction_rms']<1.2*baseline['validation_final_direction_rms'])
            trials.append(row);print(json.dumps(row),flush=True)
            key=(q['validation_restricted_matches'],-q['validation_boundary_score_rms'])
            if row['eligible']and(best is None or key>best[0]):
                best=key,rows.clone(),states.clone(),row
report={'baseline':baseline,'actual_recorded_baseline_matches':actual_baseline,
        'native_feature_count':2037,'output_coordinate_count':96,
        'native_ridge_trials':ridges,'boundary_metric_ridge_fraction_trials':metric_ridges,
        'correction_strength_trials':strengths,'training_windows':384,'fresh_validation_windows':128,
        'margin_weight_floor':.1,'maximum_token_weight':100.,
        'projection_metric':'Fixed training covariance of original HOST winner-minus-runner-up readout vectors',
        'target':'Original HOST normalized final-state difference projected into existing strong output coordinates',
        'fixed_normalizers':'Observed original A123 raw MLP RMS and final-state RMS; never updated during solves',
        'programme_G_Up_Down_and_all_gains_unchanged':True,
        'no_answers_gradients_activation_replay_or_iterative_refit':True,
        'trials':trials,'selected':None if best is None else best[3]}
out=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'projected_readout_repair.json',report)
assert best is not None
states=(norm(torch.cat([base_states,best[2]],0))*fg).float()
maxima=torch.full((len(states),),-torch.inf)
winners=torch.full((len(states),),-1,dtype=torch.long)
for start in range(0,len(E),4096):
    logits=states@E[start:start+4096].T
    values,indices=logits.max(1)
    better=values>maxima
    maxima[better]=values[better];winners[better]=start+indices[better]
nv=len(truth)
old_matches=int((winners[:nv]==truth).sum());new_matches=int((winners[nv:]==truth).sum())
report['full_vocabulary_validation']={'tokens':nv,'baseline_matches':old_matches,'selected_matches':new_matches,
                                     'exploratory_required_extra_matches':1,
                                     'prior_screen_required_extra_matches':16}
atomic_json(out.parent/'projected_readout_repair.json',report)
print(json.dumps({'full_vocabulary_validation':report['full_vocabulary_validation']}),flush=True)
assert new_matches==prior['full_vocabulary_validation']['selected_matches']
assert old_matches==prior['full_vocabulary_validation']['baseline_matches']
assert new_matches>old_matches
report['exploratory_policy']={'prior_screen_failed':True,'prior_required_gain':16,'actual_gain':13,
    'new_trials':False,'selection':'Exact immutable V1 selection, no additional search',
    'reason':'Official arithmetic and text scores determine acceptance; explicitly test this borderline fresh-calibration gain.'}
atomic_json(out.parent/'projected_readout_repair.json',report)
W=ch[dn].clone();W[pool[:,None],native[None]]=best[1]
assert torch.equal(W[:,units],ch[dn][:,units])
ch[dn]=W;c['exploratory_projected_readout_repair_a142v2']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
mask=W.view(torch.int32)!=h[dn].view(torch.int32)
nr=int(mask[:,native].sum());categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(mask.sum())-nr}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/compile_a142_projected_readout_repair_v1.py',ROOT/'experiments/derive_a142_paired_host_readout_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,tp,hp,np,fp,prior_path],
     arguments={},description=__doc__,extra=report,categories=categories)
