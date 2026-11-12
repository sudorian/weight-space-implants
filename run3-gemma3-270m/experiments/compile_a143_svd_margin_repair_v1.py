"""Larger exact HOST-margin systems over supervised native-feature SVD bases.

One original-target covariance/SVD compresses all2037 recorded native
features. Five input/output allocations then solve the exact feature-by-token
readout interaction equations, with6144--8192parameters each. Only existing
strong native Down entries change. All targets are frozen HOST predictions
on unlabeled calibration text, never arithmetic answers. No derivatives,
model/activation replay, or iterative residual fitting occurs.
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
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'
hp=ROOT/'artifacts/a56v1_host.pt'
np=ROOT/'artifacts/a140v1_host_branches.pt'
fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
w,top,oldhost,newhost,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,tp,hp,np,fp)]
assert w['checkpoint_sha256']==sha(base)and torch.equal(w['inputs'],top['inputs'])
assert torch.equal(w['inputs'][:256],oldhost['inputs'][:256])and torch.equal(w['inputs'][256:],newhost['inputs'])
for key,value in w['captures'].items():
    if key.endswith('program_features'):assert value.abs().max()==0,key
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
W0=ch[dn].double();gain=1+ch[pn].double()
units=c['units'].long();pool=c['output_pool'].long()
assert len(pool)==96 and(gain[pool]==1e6).all()
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
F=w['captures']['17.features'].double().reshape(-1,2048)
raw=w['captures']['17.down'].double().reshape(-1,640)
before=w['captures']['17.raw'].double().reshape(-1,640)
current=w['captures']['finalraw'].double().reshape(-1,640)
teacher=torch.cat([oldhost['captures']['finalraw'][:256],newhost['captures']['finalraw']],0).double().reshape(-1,640)
def norm(x):return x/(x.square().mean(1,keepdim=True)+1e-6).sqrt()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
cr=(current.square().mean(1,keepdim=True)+1e-6).sqrt()
current_n,teacher_n=norm(current),norm(teacher)
assert(F@W0.T-raw).norm()/raw.norm()<1e-5
assert(before+raw/rms*gain-current).norm()/current.norm()<1e-5
nt=384*64
host_ids=top['topk_ids'].reshape(-1,8).long();edited_ids=w['topk_ids'].reshape(-1,8).long()
expected=host_ids[:,0]
alternatives=torch.cat([host_ids[:,1:2],edited_ids[:,:2]],1)
keep=alternatives!=expected[:,None]
for j in range(1,3):keep[:,j]&=(alternatives[:,j,None]!=alternatives[:,:j]).all(1)
token,column=keep.nonzero(as_tuple=True)
alternative=alternatives[token,column]
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
D=(E[expected[token]].double()-E[alternative].double())*fg
host_gap=(teacher_n[token]*D).sum(1)
base_gap=(current_n[token]*D).sum(1)
target_gap=host_gap-base_gap
pair_weight=1/(host_gap.clamp_min(0)+.1)
training_pair=token<nt
pair_weight/=pair_weight[training_pair].mean()
pair_weight=pair_weight.clamp(max=100)
tt=token[training_pair];wt=pair_weight[training_pair];yt=target_gap[training_pair]
Dt=D[training_pair]
readout_metric=Dt.T@(Dt*wt[:,None])/len(Dt)
me,mv=torch.linalg.eigh((readout_metric+readout_metric.T)/2)
metric_root=mv*me.clamp_min(0).sqrt()[None]
pool_metric=readout_metric[pool][:,pool]
pe,pv=torch.linalg.eigh((pool_metric+pool_metric.T)/2)
output_basis=pv[:,pe.argsort(descending=True)]
output_energy=pe[pe.argsort(descending=True)].clamp_min(0)
gap=(top['topk_values'].reshape(-1,8)[:,0]-top['topk_values'].reshape(-1,8)[:,1]).double().clamp_min(0)
token_weight=1/(gap[:nt]+.1);token_weight/=token_weight.mean();token_weight=token_weight.clamp(max=100)
X=F[:nt,native]/rms[:nt]/cr[:nt]*token_weight[:,None].sqrt()
sd=X.square().mean(0).sqrt().clamp_min(1e-15)
Z=X/sd
C=Z.T@Z/nt
fe,fv=torch.linalg.eigh((C+C.T)/2);fe=fe.clamp_min(0)
feature_ridge=.001
whitener=fv/(fe+feature_ridge).sqrt()[None]
cross=Z.T@((teacher_n[:nt]-current_n[:nt])*token_weight[:,None].sqrt())/nt
whitened_cross=whitener.T@cross@metric_root
left,singular,_=torch.linalg.svd(whitened_cross,full_matrices=False)
input_basis=(whitener@left[:,:512])/sd[:,None]
input_features=(F[:,native]/rms/cr)@input_basis
del X,Z,C,fe,fv,whitener,cross,whitened_cross,left,Dt,metric_root
validation_ids=torch.cat([host_ids[nt:],edited_ids[nt:]],1)
rivals=E[validation_ids].double()*fg[None,None]
truth=expected[nt:]
validation_pair_tokens=token[~training_pair]-nt
VD=D[~training_pair];VW=pair_weight[~training_pair];VG=host_gap[~training_pair]
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
PB=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
PR=PF@W0.T
format_reference=norm(fmt['captures']['finalraw'][:,8:13].double().reshape(-1,640))
format_margin=(fmt['topk_values'][:,:,0]-fmt['topk_values'][:,:,1]).double().reshape(-1)
readout_norm=0.
for start in range(0,len(E),4096):readout_norm=max(readout_norm,float((E[start:start+4096].double()*fg).norm(dim=1).max()))
def quality(delta):
    rr=raw[nt:].clone();rr[:,pool]+=F[nt:,native]@delta.T
    rrms=(rr.square().mean(1,keepdim=True)+1e-6).sqrt()
    states=before[nt:]+rr/rrms*gain
    scores=torch.einsum('nd,nkd->nk',states,rivals)
    winners=validation_ids.gather(1,scores.argmax(1,keepdim=True)).flatten()
    score_gap=((states/cr[nt:])[validation_pair_tokens]*VD).sum(1)
    error=score_gap-VG
    fr=PR.clone();fr[:,pool]+=PF[:,native]@delta.T
    final=norm(PB+norm(fr)*gain)
    bound=float((format_margin-2*(final-format_reference).norm(dim=1)*readout_norm).min())
    return {'validation_restricted_matches':int((winners==truth).sum()),
            'validation_weighted_margin_rms':float((error.square()*VW).sum().sqrt()/VW.sum().sqrt()),
            'validation_final_direction_rms':float((norm(states)-teacher_n[nt:]).square().mean().sqrt()),
            'maximum_relative_raw_norm_change':float((rrms/rms[nt:]-1).abs().max()),
            'minimum_recorded_format_margin_bound':bound},states
baseline,base_states=quality(torch.zeros(len(pool),len(native),dtype=torch.float64))
assert abs(baseline['validation_restricted_matches']-int((edited_ids[nt:,0]==truth).sum()))<=2
allocations=[(64,96),(96,80),(128,64),(256,32),(512,16)]
ridges=[.01,.1,1.,10.,100.]
trials,best=[],None
out=ROOT/'experiments/a143v1_svd_margin_repair/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({'baseline':baseline,'training_pairs':len(tt),'allocations':allocations}),flush=True)
for columns,rank in allocations:
    B=output_basis[:,:rank]
    projected_readout=((E[expected[tt]].double()-E[alternative[training_pair]].double())*fg)[:,pool]@B
    npar=columns*rank
    gram=torch.zeros(npar,npar,dtype=torch.float64)
    rhs=torch.zeros(npar,dtype=torch.float64)
    for start in range(0,len(tt),256):
        s=slice(start,min(start+256,len(tt)))
        interaction=(input_features[tt[s],:columns,None]*projected_readout[s,None,:]).reshape(-1,npar)
        weighted=interaction*wt[s,None].sqrt()
        gram+=weighted.T@weighted/len(tt)
        rhs+=interaction.T@(wt[s]*yt[s])/len(tt)
    scale=gram.diagonal().sqrt().clamp_min(1e-15)
    gram/=scale[:,None]*scale[None,:];rhs/=scale
    print(json.dumps({'allocation':[columns,rank],'normal_equations_parameters':npar,'stage':'eigendecomposition'}),flush=True)
    ev,V=torch.linalg.eigh((gram+gram.T)/2);ev=ev.clamp_min(0)
    projected_rhs=V.T@rhs
    for ridge in ridges:
        beta=(V@(projected_rhs/(ev+ridge)))/scale
        theta=beta.reshape(columns,rank)
        correction=(input_basis[:,:columns]@theta@B.T).T/gain[pool,None]
        rows=(W0[pool[:,None],native[None]]+correction).float()
        delta=rows.double()-W0[pool[:,None],native[None]]
        q,states=quality(delta)
        row={'input_basis_rank':columns,'output_basis_rank':rank,'parameters':npar,'ridge':ridge,
             'retained_output_metric_energy':float(output_energy[:rank].sum()/output_energy.sum()),**q,
             'maximum_native_weight':float(rows.abs().max())}
        row['eligible']=bool(rows.isfinite().all()and rows.abs().max()<=1e6 and
            q['maximum_relative_raw_norm_change']<.001 and q['minimum_recorded_format_margin_bound']>1.
            and q['validation_final_direction_rms']<1.2*baseline['validation_final_direction_rms'])
        trials.append(row);print(json.dumps(row),flush=True)
        key=(q['validation_restricted_matches'],-q['validation_weighted_margin_rms'])
        if row['eligible']and(best is None or key>best[0]):best=key,rows.clone(),states.clone(),row
    del gram,rhs,ev,V,projected_rhs,projected_readout
    atomic_json(out.parent/'partial_trials.json',{'baseline':baseline,'trials':trials,'selected':None if best is None else best[3]})
report={'baseline':baseline,'feature_basis_allocations':[list(v)for v in allocations],
        'ridge_trials':ridges,'feature_whitening_ridge':feature_ridge,'margin_weight_floor':.1,
        'maximum_training_weight':100.,'training_windows':384,'fresh_validation_windows':128,
        'input_basis':'One whitened covariance/SVD of the original paired HOST final-state difference in the fixed boundary metric',
        'input_source_features':2037,'training_pairs':len(tt),'validation_pairs':len(VD),
        'pair_rule':'HOST runner-up and edited top two, omitting winner and duplicates',
        'programme_weights_G_Up_and_all_gains_unchanged':True,
        'no_answers_gradients_activation_replay_or_iterative_refit':True,
        'trials':trials,'selected':None if best is None else best[3]}
atomic_json(out.parent/'svd_margin_repair.json',report)
assert best is not None
states=(norm(torch.cat([base_states,best[2]],0))*fg).float()
maxima=torch.full((len(states),),-torch.inf);winners=torch.full((len(states),),-1,dtype=torch.long)
for start in range(0,len(E),4096):
    logits=states@E[start:start+4096].T
    values,indices=logits.max(1);improved=values>maxima
    maxima[improved]=values[improved];winners[improved]=start+indices[improved]
nv=len(truth);old_matches=int((winners[:nv]==truth).sum());new_matches=int((winners[nv:]==truth).sum())
report['full_vocabulary_validation']={'tokens':nv,'baseline_matches':old_matches,'selected_matches':new_matches,'minimum_required_extra_matches':8}
atomic_json(out.parent/'svd_margin_repair.json',report)
print(json.dumps({'full_vocabulary_validation':report['full_vocabulary_validation']}),flush=True)
assert new_matches>=old_matches+8,'Less than eight additional fresh validation matches'
W=ch[dn].clone();W[pool[:,None],native[None]]=best[1]
assert torch.equal(W[:,units],ch[dn][:,units])
ch[dn]=W;c['svd_margin_repair_a143']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
mask=W.view(torch.int32)!=h[dn].view(torch.int32);nr=int(mask[:,native].sum())
categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(mask.sum())-nr}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a142_paired_host_readout_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,tp,hp,np,fp],
     arguments={},description=__doc__,extra=report,categories=categories)
