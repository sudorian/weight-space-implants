"""Recorded-state feasibility screen for additional zero-output RMS carriers.

This is an optimistic diagnostic, not deployable weights. Select quiet
ordinary rows using original training text, set their proposed output gain
to zero, and analytically optimize the remaining branch's radial factor.
No native activation or model forward pass is executed.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
ap=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt';op=ROOT/'artifacts/a56v1_host.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'
ch,ref=expand(h,base),expand(h,reference)
a,host,oldhost,top=[torch.load(p,weights_only=True,mmap=True) for p in (ap,hp,op,tp)]
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn,pn):assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
assert a['checkpoint_sha256']==sha(reference)
assert torch.equal(a['inputs'],top['inputs'])
assert torch.equal(a['inputs'][:256],oldhost['inputs'][:256]) and torch.equal(a['inputs'][256:],host['inputs'])
c=torch.load(base.parent/'circuit.pt',weights_only=True);units=c['units'].long()
F=a['captures']['17.features'].reshape(-1,2048).double();assert F[:,units].abs().max()==0
before=a['captures']['17.raw'].reshape(-1,640).double()
W=ch[dn].double();gain=1+ch[pn].double();raw=F@W.T
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
rr=rms(raw);parent=before+raw/rr*gain
recorded=before+norm(F@ref[dn].double().T)*(1+ref[pn].double())
reconstruction=float((recorded-a['captures']['finalraw'].reshape(-1,640).double()).norm()/recorded.norm())
assert reconstruction<1e-6
del F,recorded,ref
teacher=torch.cat([oldhost['captures']['finalraw'][:256],host['captures']['finalraw']]).reshape(-1,640).double()
tr=rms(teacher);desired=teacher-before;numerator=raw*gain
nt=384*64
allowed=~W[:,units].ne(0).any(1)
importance=((numerator[:nt]/rr[:nt]/tr[:nt]).square()).mean(0)
importance[~allowed]=torch.inf
order=importance.argsort();counts=[1,2,4,8,16]
assert int(allowed.sum())>=max(counts)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
val_ids=torch.cat([top['topk_ids'].reshape(-1,8)[nt:],a['topk_ids'].reshape(-1,8)[nt:]],1).long()
truth=top['topk_ids'].reshape(-1,8)[nt:,0].long()
def quality(state):
    matches=0
    for start in range(0,len(state),256):
        ids=val_ids[start:start+256]
        vectors=E[ids].double()*fg
        score=torch.einsum('nd,nkd->nk',state[start:start+256],vectors)
        prediction=ids.gather(1,score.argmax(1,keepdim=True)).flatten()
        matches+=int((prediction==truth[start:start+256]).sum())
    return {'validation_restricted_matches':matches,
        'validation_paired_HOST_state_relative_rms':float(((state-teacher[nt:])/tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms':float((norm(state)-norm(teacher[nt:])).square().mean().sqrt())}
baseline=quality(parent[nt:]);trials=[]
for count in counts:
    anchors=order[:count]
    remaining_numerator=numerator.clone();remaining_numerator[:,anchors]=0
    cross=(remaining_numerator*desired).sum(1)
    energy=remaining_numerator.square().sum(1)
    assert energy.min()>0
    alpha_unconstrained=cross/energy
    outside_energy=(raw.square().sum(1)-raw[:,anchors].square().sum(1)).clamp_min(0)
    lower=.5/rr[:,0]
    upper=torch.minimum(2/rr[:,0],(640/(outside_energy+640e-6)).sqrt())
    assert (upper>=lower).all()
    alpha=torch.maximum(lower,torch.minimum(alpha_unconstrained,upper))
    state=before+alpha[:,None]*remaining_numerator
    q=quality(state[nt:])
    room=(640/alpha.square()-outside_energy-640e-6).clamp_min(0)
    rec={'anchor_count':count,'anchors':anchors.tolist(),'old_output_gains':gain[anchors].tolist(),**q,
        'training_lower_caps':int((alpha_unconstrained[:nt]<lower[:nt]).sum()),
        'training_upper_caps':int((alpha_unconstrained[:nt]>upper[:nt]).sum()),
        'validation_lower_caps':int((alpha_unconstrained[nt:]<lower[nt:]).sum()),
        'validation_upper_caps':int((alpha_unconstrained[nt:]>upper[nt:]).sum()),
        'mean_training_required_raw_anchor_energy_in_parent_RMS_units':float((room[:nt]/rr[:nt,0].square()).mean()),
        'maximum_validation_radial_RMS_relative_change':float((1/(alpha[nt:]*rr[nt:,0])-1).abs().max())}
    rec['promising_oracle_only']=bool(q['validation_restricted_matches']>=baseline['validation_restricted_matches']+16
        and q['validation_paired_HOST_state_relative_rms']<baseline['validation_paired_HOST_state_relative_rms']
        and q['validation_paired_HOST_direction_rms']<baseline['validation_paired_HOST_direction_rms'])
    trials.append(rec);print(json.dumps(rec),flush=True)
report={'analysis_only':True,'no_deployable_weights':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'training_windows':384,'validation_windows':128,'anchor_count_trials':counts,
    'fixed_proposed_anchor_output_gain':0,'radial_inverse_RMS_domain_relative_to_parent':[.5,2.],
    'row_selection':'Lowest original training post-branch output energy relative to HOST state RMS; every nonzero arithmetic programme output row excluded',
    'target':'Analytic positive radial least-squares projection of original paired-HOST final state with selected output coordinates zeroed, capped by unchanged raw coordinate energy',
    'oracle_scope':'Uses original HOST state per validation token solely for an optimistic feasibility diagnostic; no deployable function or accuracy claim. These per-token factors are never installed.',
    'baseline':baseline,'trials':trials,'any_promising':any(r['promising_oracle_only'] for r in trials),
    'parent_recorded_state_reconstruction_relative_error':reconstruction,
    'no_gradients_native_activation_execution_answers_or_iterative_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',reference,ap,hp,op,tp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a236v1_zero_gain_radial_oracle.json',report)
print(json.dumps({'any_promising':report['any_promising'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
