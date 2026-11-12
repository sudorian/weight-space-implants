"""Restore final native RMS through a fitted norm-envelope anchor.

The nearly silent existing coordinate439 carries the square-root energy
missing after96strong writer rows are attenuated. Its target is an original
HOST native raw-energy identity on unlabeled text, not a sum. This differs
from earlier PCA/trace carriers and row-scale energy fits: a single fixed
nonlinear energy target is projected onto recorded native features once.
Programme columns, G/Up, strong gains and the raw programme anchor stay fixed.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha,ELIGIBLE
h=host_state()
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
cp=ROOT/'experiments/a122v1_native17/delta.pt'
bp=ROOT/'artifacts/a122v1_final_basis.pt'
np=ROOT/'artifacts/a122v1_native8_word.pt'
tp=ROOT/'artifacts/a83v1_host_top.pt'
fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch,reference=expand(h,base),expand(h,cp)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
b,n8,top,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(bp,np,tp,fp)]
assert b['checkpoint_sha256']==sha(cp)
assert torch.equal(b['inputs'][:384],top['inputs'][:384])
gn,un,dn,pn=[f'model.layers.17.{s}'for s in(
    'mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
for name in h:
    if name in(gn,un,dn,pn):continue
    if name in('model.layers.8.mlp.gate_proj.weight','model.layers.8.mlp.down_proj.weight'):continue
    assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
assert b['captures']['word.8.program_features'].abs().max()==0
u8=torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert (n8['captures']['8.pre'][:384].double().reshape(-1,640)@ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max()<-16
units=c['units'].long();pool=c['output_pool'].long();anchor=439
assert len(units)==11 and len(pool)==96 and anchor not in pool.tolist()
assert (b['captures']['word.17.pre'].double().reshape(-1,640)@ch[gn][units].double().T).max()<-32
FH=b['captures']['word.17.features'].double().reshape(-1,2048)
WH=h[dn].double();gh=1+h[pn].double();gb=1+ch[pn].double()
F=FH.clone();F[:,units]=0
RH=FH@WH.T
hr=(RH.square().mean(1,keepdim=True)+1e-6).sqrt()
target=RH/hr*gh
nt=256*64
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
donors=c['native_donor_repairs']['17']['donors'].long()
assert len(donors)==512 and not torch.isin(donors,units).any()
X=FH[:nt,donors]/hr[:nt]
sd=X.square().mean(0).sqrt().clamp_min(1e-12)
Z=X/sd
C=Z.T@Z/len(Z);C.diagonal().add_(.001)
repair=torch.linalg.solve(C,Z.T@(FH[:nt,units]/hr[:nt])/len(Z))/sd[:,None]
W0=WH.clone()
W0[:,donors]+=WH[:,units]@repair.T
W0[pool]*=(gh[pool]/gb[pool])[:,None]
W0[:,units]=ch[dn][:,units].double()
# Full surviving features reconstruct original HOST strong-row numerators
# at the fixed HOST normalizer, before deriving the single energy target.
XN=F[:,native]/hr
sdn=XN[:nt].square().mean(0).sqrt().clamp_min(1e-12)
ZN=XN[:nt]/sdn
C=ZN.T@ZN/nt
ev,V=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
rhs=V.T@(ZN.T@(target[:nt,pool]/gb[pool])/nt)
ridges=[1e-6,1e-5,1e-4,.001,.01]
strong_trials=[];strong_best=None
for ridge in ridges:
    fit=(V@(rhs/(ev[:,None]+ridge)))/sdn[:,None]
    score=float(((XN[nt:]@fit)*gb[pool]-target[nt:,pool]).square().mean().sqrt())
    strong_trials.append({'ridge':ridge,'validation_fixed_HOST_normalizer_strong_rms':score})
    if strong_best is None or score<strong_best[0]:strong_best=score,fit.clone(),ridge
W0[pool[:,None],native[None]]=strong_best[1].T
W0=W0.float().double()
R0=F@W0.T
other_energy=R0.square().sum(1)-R0[:,anchor].square()
room=RH.square().sum(1)-other_energy
feasible=room>=0
sign=1. if float(RH[:nt,anchor].mean())>=0 else -1.
anchor_target=sign*room.clamp_min(0).sqrt()
anchor_rhs=V.T@(ZN.T@(anchor_target[:nt,None]/hr[:nt])/nt)
parameter=h[pn].clone()
parameter[pool]=ch[pn][pool]
parameter[anchor]=ch[pn][anchor]
gain=1+parameter.double()
before=b['captures']['word.17.raw'][256:].double().reshape(-1,640)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
truth=top['topk_ids'][256:384,:,0].reshape(-1)
v_ids=top['topk_ids'][256:384].reshape(-1,8).long()
rivals=E[v_ids].double()*fg[None,None]
def normalized(x):return x/(x.square().mean(1,keepdim=True)+1e-6).sqrt()
def quality(W,g):
    R=F[nt:]@W.double().T
    post=normalized(R)*g
    state=before+post
    scores=torch.einsum('nd,nkd->nk',state,rivals)
    return {'validation_native_branch_rms':float((post-target[nt:]).square().mean().sqrt()),
            'validation_relative_raw_rms_error':float(((R.square().mean(1,keepdim=True)+1e-6).sqrt()/hr[nt:]-1).square().mean().sqrt()),
            'validation_HOST_top8_matches':int((scores.argmax(1)==0).sum())},state
baseline,base_states=quality(ch[dn],gb)
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
PB=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
format_old=normalized(fmt['captures']['finalraw'][:,8:13].double().reshape(-1,640))
format_margin=(fmt['topk_values'][:,:,0]-fmt['topk_values'][:,:,1]).double().reshape(-1)
max_readout_norm=0.
for start in range(0,len(E),4096):
    max_readout_norm=max(max_readout_norm,float((E[start:start+4096].double()*fg).norm(dim=1).max()))
bm=json.loads(base.with_suffix('.manifest.json').read_text())
trials,best=[],None
print(json.dumps({'baseline':baseline,'anchor':anchor,'host_anchor_gain':float(gh[anchor]),
                  'anchor_sign':sign,'infeasible_training_rows':int((~feasible[:nt]).sum()),
                  'infeasible_validation_rows':int((~feasible[nt:]).sum()),
                  'strong_trials':strong_trials}),flush=True)
for ridge in ridges:
    fit=(V@(anchor_rhs/(ev[:,None]+ridge)))/sdn[:,None]
    fitted=XN[:nt]@fit
    # One analytic training energy match, not a residual refit.
    trace_scale=float(((anchor_target[:nt,None]/hr[:nt]).square().sum()/fitted.square().sum()).sqrt())
    W=W0.clone();W[anchor,native]=fit[:,0]*trace_scale
    Wf=W.float();assert torch.equal(Wf[:,units],ch[dn][:,units])
    q,state=quality(Wf,gain)
    format_new=normalized(PB+normalized(PF@Wf.double().T)*gain)
    bound=float((format_margin-2*(format_new-format_old).norm(dim=1)*max_readout_norm).min())
    count=bm['changed_scalars']-bm['per_tensor'][dn]-bm['per_tensor'][pn]
    count+=int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
    count+=int((parameter.view(torch.int32)!=h[pn].view(torch.int32)).sum())
    row={'anchor_ridge':ridge,'analytic_energy_trace_scale':trace_scale,**q,
         'minimum_recorded_format_margin_bound':bound,'changed_scalars':count,
         'maximum_down_weight':float(Wf.abs().max())}
    row['eligible']=bool(count<=int(.05*ELIGIBLE)and Wf.isfinite().all()and Wf.abs().max()<=1e6 and bound>1.)
    trials.append(row);print(json.dumps(row),flush=True)
    if row['eligible']and(best is None or q['validation_native_branch_rms']<best[0]):
        best=q['validation_native_branch_rms'],Wf.clone(),state.clone(),row
report={'baseline':baseline,'anchor':anchor,'anchor_sign':sign,'donor_count':512,'donor_ridge':.001,
        'strong_ridge_trials':ridges,'strong_trials':strong_trials,'selected_strong_ridge':strong_best[2],
        'anchor_ridge_trials':ridges,'trials':trials,'selected':None if best is None else best[3],
        'infeasible_training_rows':int((~feasible[:nt]).sum()),'infeasible_validation_rows':int((~feasible[nt:]).sum()),
        'infeasible_policy':'Clip negative energy remainder to zero; all validation rows still count',
        'target':'Square-root original HOST native energy remainder at the exact current word input',
        'ordinary_gains':'Reset to HOST, except the existing programme raw anchor gain',
        'programme_G_Up_Down_and_writer_gains_fixed':True,
        'no_answers_gradients_activation_replay_or_iterative_refit':True}
out=ROOT/'experiments/a141v1_nonlinear_energy_anchor/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'energy_anchor.json',report)
assert best is not None and best[0]<.9*baseline['validation_native_branch_rms'],'Less than ten percent local branch improvement'
states=torch.cat([base_states,best[2]],0)
states=(normalized(states)*fg).float()
maxima=torch.full((len(states),),-torch.inf)
winners=torch.full((len(states),),-1,dtype=torch.long)
for start in range(0,len(E),4096):
    logits=states@E[start:start+4096].T
    values,indices=logits.max(1)
    improved=values>maxima
    maxima[improved]=values[improved];winners[improved]=start+indices[improved]
nv=len(truth)
old_matches=int((winners[:nv]==truth).sum());new_matches=int((winners[nv:]==truth).sum())
report['full_vocabulary_validation']={'tokens':nv,'baseline_matches':old_matches,'new_matches':new_matches}
atomic_json(out.parent/'energy_anchor.json',report)
print(json.dumps({'full_vocabulary_validation':report['full_vocabulary_validation']}),flush=True)
assert new_matches>=old_matches,'Full-vocabulary calibration agreement decreased'
ch[dn],ch[pn]=best[1],parameter
c['nonlinear_energy_anchor_a141']=report
atomic_torch(out.parent/'circuit.pt',c)
categories=dict(bm['categories'])
mask=ch[dn].view(torch.int32)!=h[dn].view(torch.int32)
nr=int(mask[:,native].sum());categories[dn]={'native_text_repair_and_energy_anchor':nr,'arithmetic_programme':int(mask.sum())-nr}
mask=parameter.view(torch.int32)!=h[pn].view(torch.int32)
categories[pn]={'arithmetic_writer_gains':int(mask[pool].sum()),'native_text_repair':int(mask.sum())-int(mask[pool].sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',cp,bp,np,tp,fp],
     arguments={},description=__doc__,extra=report,categories=categories)
