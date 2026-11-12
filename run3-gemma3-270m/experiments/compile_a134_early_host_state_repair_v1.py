"""One original-HOST early-state reconstruction at the receiver-free input.

Programme weights and all gains remain fixed. Each direct solve starts from
HOST Down with fixed gain accommodation and targets the paired original HOST3
state. The reference normalizer is fixed before fitting. Five weak-anchor counts
are assessed using original-target feasibility and ideal state error, before
any fitted weight map is evaluated. Format perturbations
are bounded from recorded native features and the uniform hand-circuit
GELU-to-ReLU error bound; no nonlinear activation is replayed.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha,ELIGIBLE
h=host_state();base=ROOT/'experiments/a132v1_local_selector_repair/delta.pt'
cp=ROOT/'experiments/a133v1_native3/delta.pt'
wp=ROOT/'artifacts/a133v1_native3_basis.pt';hp=ROOT/'artifacts/a133v1_original_host.pt'
ch,reference=expand(h,base),expand(h,cp)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
w,host=[torch.load(p,weights_only=True,mmap=True)for p in(wp,hp)]
assert w['checkpoint_sha256']==sha(cp)
assert torch.equal(w['inputs'][:384],host['inputs'][:384])
prefix='model.layers.3.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight',
                              'mlp.down_proj.weight','post_feedforward_layernorm.weight')]
for name in h:
    if any(name.startswith(f'model.layers.{i}.')for i in range(3))or(
        name.startswith(prefix)and name not in(gn,un,dn,pn)):
        assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
for name in(gn,un,dn,pn):assert torch.equal(reference.get(name,h[name]),h[name]),name
for name,value in w['captures'].items():
    if name.endswith('program_features'):assert value.abs().max()==0,name
programme=c['native_a_stage']['units'].long()
assert len(programme)==4
Wcurrent=ch[dn].double();Wprog=Wcurrent[:,programme]
parameter=ch[pn];gamma=1+parameter.double();gh=1+h[pn].double()
F=w['captures']['word.3.features'].double().reshape(-1,2048);F[:,programme]=0
pre=w['captures']['word.3.pre'].double().reshape(-1,640)
maximum_gate=float((pre@ch[gn][programme].double().T).max());assert maximum_gate<-16
before=w['captures']['word.3.raw'].double().reshape(-1,640)
raw=F@Wcurrent.T;rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
teacher=host['captures']['3.out'][:384].double().reshape(-1,640)
ref=(teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
nt=256*64;training=torch.arange(len(F))<nt;validation=~training
Worigin=h[dn].double().clone();strong=(gamma.abs()>1e5).nonzero().flatten();assert len(strong)==3
Worigin[strong]*=(gh[strong]/gamma[strong])[:,None];Worigin[:,programme]=Wprog
origin_raw=F@Worigin.T;origin_z=origin_raw/rms
Fn=F/rms/ref;std=Fn[:nt].square().mean(0).sqrt().clamp_min(1e-12)
Z=Fn[:nt]/std
zero_gain=gamma==0
literal_target=(teacher-before)/torch.where(zero_gain,torch.ones_like(gamma),gamma)
literal_target[:,zero_gain]=origin_z[:,zero_gain]
priority=(Z.T@((literal_target-origin_z)/ref)[:nt]/nt).square().sum(1);priority[programme]=-torch.inf
fixed=c['native_a_stage']['donors'].long().tolist();assert not set(fixed)&set(programme.tolist())
order=fixed+[i for i in priority.argsort(descending=True).tolist()if i not in set(fixed)|set(programme.tolist())]
counts=[512,768,1024,1536,2044];ridges=[1e-4,.001,.01,.1,1.]
allowed=~Wprog.ne(0).any(1);cost=gamma.abs().clone();cost[~allowed]=torch.inf
geometry=[];choices=[]
for count in(8,16,32,64,128):
    anchor=cost.argsort()[:count];assert allowed[anchor].all()
    candidate=literal_target.clone();candidate[:,anchor]=0
    radius=640*(1-1e-6/rms[:,0].square()).clamp_min(0)
    room=radius-candidate.square().sum(1);ok=room>0
    direction=origin_z[:,anchor];direction/=direction.norm(dim=1,keepdim=True).clamp_min(1e-30)
    candidate[:,anchor]=direction*room.clamp_min(0).sqrt()[:,None]
    ideal=before+candidate*gamma
    error=(ideal-teacher)/ref
    row={'anchor_count':count,'training_feasible_fraction':float(ok[training].double().mean()),
         'infeasible_rows':int((~ok).sum()),
         'ideal_feasible_state_rms':float(error[ok&validation].square().mean().sqrt())}
    row['eligible']=row['training_feasible_fraction']>.95
    geometry.append(row)
    if row['eligible']:choices.append((row['ideal_feasible_state_rms'],candidate,anchor,ok,row))
assert choices,'No adequate original-target geometry'
_,target,anchors,feasible,chosen_geometry=min(choices,key=lambda x:x[0])
anchor_count=len(anchors);used=training&feasible
print(json.dumps({'target_geometry':geometry,'selected_geometry':chosen_geometry}),flush=True)
desired=(target-origin_z)/ref;Zu=Fn[used]/std
Cu=Zu.T@Zu/len(Zu);rhs=Zu.T@desired[used]/len(Zu)

# Native nonprogramme features are exact official observations. Only the
# explicitly hand-written four features use a ReLU ideal plus a uniform bound.
PF=w['captures']['format.3.features'][:,8:9].double().reshape(-1,2048)
PP=w['captures']['format.3.pre'][:,8:9].double().reshape(-1,640)
PG=PP@ch[gn][programme].double().T;PU=PP@ch[un][programme].double().T
PF[:,programme]=PG.clamp_min(0)*PU
PR=PF@Wcurrent.T
error_raw=.25*PU.abs()@Wprog.abs().T
error_norm=error_raw.norm(dim=1)
lower=(PR.square().sum(1)+640e-6).sqrt()-error_norm
assert lower.min()>0

def quality(pred):
    hidden=before[validation]+pred/(pred.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    relative=(hidden-teacher[validation])/ref[validation]
    direction=hidden/(hidden.square().mean(1,keepdim=True)+1e-6).sqrt()-teacher[validation]/ref[validation]
    return {'relative_hidden_state_rms':float(relative.square().mean().sqrt()),
            'hidden_direction_rms':float(direction.square().mean().sqrt()),
            'relative_token_error_p99':float(relative.square().mean(1).sqrt().quantile(.99))}
baseline=quality(raw[validation]);print(json.dumps({'baseline':baseline}),flush=True)
bm=json.loads(base.with_suffix('.manifest.json').read_text());other_count=bm['changed_scalars']-bm['per_tensor'][dn]
trials=[];best=None
for count in counts:
    columns=torch.tensor(order[:count]);gram=Cu[columns][:,columns]
    ev,evec=torch.linalg.eigh((gram+gram.T)/2);ev=ev.clamp_min(0);projected=evec.T@rhs[columns]
    for ridge in ridges:
        delta=(evec@(projected/(ev[:,None]+ridge)))/std[columns,None]
        W=Worigin.clone();W[:,columns]+=delta.T;Wf=W.float();W=Wf.double()
        assert torch.equal(Wf[:,programme],Wprog.float())
        q=quality(F[validation]@W.T)
        difference=PF@(W-Wcurrent).T;d=difference.norm(dim=1)
        if (lower>d).all():
            bound=640**.5*gamma.abs()*(difference.abs()/(lower-d)[:,None]+
                (PR.abs()+error_raw)*(d/(lower*(lower-d)))[:,None])
            format_bound=float(bound.max())
        else:format_bound=float('inf')
        changed=other_count+int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
        row={'anchor_count':anchor_count,'donor_count':count,'ridge':ridge,'training_rows':int(used.sum()),
             'infeasible_rows':int((~feasible).sum()),**q,'maximum_format_post_change_bound':format_bound,
             'changed_scalars':changed,'maximum_down_weight':float(Wf.abs().max())}
        row['eligible']=bool(len(Zu)>.95*nt and changed<=int(.05*ELIGIBLE)and Wf.isfinite().all()
            and Wf.abs().max()<=1e6 and format_bound<100 and q['relative_token_error_p99']<1.1*baseline['relative_token_error_p99'])
        trials.append(row);print(json.dumps(row),flush=True)
        if row['eligible']and(best is None or q['relative_hidden_state_rms']<best[0]):
            best=q['relative_hidden_state_rms'],Wf.clone(),row
out=ROOT/'experiments/a134v1_early_host_state_repair/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
report={'anchor_count':anchor_count,'anchor_count_trials':[8,16,32,64,128],'target_geometry':geometry,
        'selected_geometry':chosen_geometry,'zero_gain_rows':zero_gain.nonzero().flatten().tolist(),
        'donor_count_trials':counts,'ridge_trials':ridges,
        'training_windows':256,'validation_windows':128,'format_windows':128,'format_bound_positions':[8],
        'maximum_word_programme_gate':maximum_gate,'baseline':baseline,'trials':trials,
        'selected':None if best is None else best[2],
        'target':'Paired original HOST3 states on unlabeled text; each fit starts from HOST Down.',
        'reference_normalizer_fixed_before_all_fits':True,'programme_and_all_gains_unchanged':True,
        'format_bound':'Recorded native features, uniform |GELU(g)-ReLU(g)| <= 0.25, coordinatewise RMS perturbation bound.',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'early_host_state_repair.json',report)
assert best is not None and best[0]<.95*baseline['relative_hidden_state_rms'],'Less than five-percent current-state improvement'
ch[dn]=best[1];c['early_host_state_repair_a134']=report
atomic_torch(out.parent/'circuit.pt',c)
categories=dict(bm['categories']);changed=ch[dn].view(torch.int32)!=h[dn].view(torch.int32)
mask=torch.ones_like(changed);mask[:,programme]=False;nr=int((changed&mask).sum())
categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(changed.sum())-nr}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',cp,wp,hp,
             ROOT/'artifacts/a133v1_native3_basis_layout.json'],arguments={},description=__doc__,extra=report,categories=categories)
