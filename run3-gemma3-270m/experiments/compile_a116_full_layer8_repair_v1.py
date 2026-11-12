"""Independent original-HOST state repair using the expanded layer8 native basis.

Every solve starts from HOST Down weights with only fixed writer-gain
accommodation and programme columns inserted. Targets are paired original
HOST layer8 states on unlabeled calibration. The previous global layer8 fit
is not a regression starting point. Its observed raw RMS is a fixed design
reference; it is never updated using fitted outputs. No activation replay,
gradients, arithmetic answer targets or iterative residual fits occur.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,ELIGIBLE
p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
wp=ROOT/'artifacts/a115v1_native_word.pt';hp=ROOT/'artifacts/a73v1_host8_only.pt'
fp=ROOT/'artifacts/a116v1_layer8_format.pt';refbase=ROOT/'experiments/a114v1_shared_phase_frame/delta.pt'
reference_ch=expand(h,refbase)
for name in h:
    if any(name.startswith(f'model.layers.{layer}.')for layer in range(9)):
        assert torch.equal(ch.get(name,h[name]),reference_ch.get(name,h[name])),name
del reference_ch
w,host,fmt=[torch.load(x,weights_only=True,mmap=True)for x in(wp,hp,fp)]
assert torch.equal(w['inputs'],host['inputs'])
prefix='model.layers.8.'
gn,un,dn,pn=[prefix+x for x in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
programme=((ch[gn]!=h[gn]).any(1)|(ch[un]!=h[un]).any(1)).nonzero().flatten()
F=w['captures']['8.features'].double().reshape(-1,2048)
assert F[:,programme].abs().max()==0
before=w['captures']['8.raw'].double().reshape(-1,640)
raw=w['captures']['8.down'].double().reshape(-1,640)
original=w['captures']['8.out'].double().reshape(-1,640)
teacher=host['captures']['8.out'].double().reshape(-1,640)
Wcurrent=ch[dn].double();gamma=1+ch[pn].double();gh=1+h[pn].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
assert (before+raw/rms*gamma-original).abs().max()<.01
ref=(teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
nt=384*64;training=torch.arange(len(F))<nt;validation=~training
surviving=torch.arange(2048)[~torch.isin(torch.arange(2048),programme)]
assert len(surviving)==2037

# A fixed HOST-based starting map, distinct from every previous fitted map.
Worigin=h[dn].double().clone()
strong=(gamma.abs()>1e5).nonzero().flatten()
Worigin[strong]*=(gh[strong]/gamma[strong])[:,None]
Worigin[:,programme]=Wcurrent[:,programme]
origin_raw=F@Worigin.T;origin_z=origin_raw/rms
Fn=F/rms/ref
std=Fn[:nt].square().mean(0).sqrt().clamp_min(1e-12)
Z=Fn[:nt]/std
literal_target=(teacher-before)/gamma
priority=(Z.T@((literal_target-origin_z)/ref)[:nt]/nt).square().sum(1)
priority[programme]=-torch.inf
fixed=c['native_donor_repairs']['8']['donors'].long().tolist()
order=fixed+[i for i in priority.argsort(descending=True).tolist()if i not in set(fixed)|set(programme.tolist())]
counts=[512,768,1024,1536,2037];ridges=[1e-4,.001,.01,.1,1.];anchor_counts=[1,2,4,8,16]
allowed=~Wcurrent[:,programme].ne(0).any(1)
anchor_cost=gamma.abs().clone();anchor_cost[~allowed]=torch.inf
anchor_order=anchor_cost.argsort()[:16]
assert allowed[anchor_order].all()
PF=fmt['captures']['8.features'][:,8:13].double().reshape(-1,2048)
PR=fmt['captures']['8.down'][:,8:13].double().reshape(-1,640)
format_post0=PR/(PR.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
format_reconstruction_relative=float((PF@Wcurrent.T-PR).norm()/PR.norm())
assert format_reconstruction_relative<1e-6
def quality(pred):
    post=pred/(pred.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    hidden=before[validation]+post
    relative=(hidden-teacher[validation])/ref[validation]
    direction=hidden/(hidden.square().mean(1,keepdim=True)+1e-6).sqrt()-teacher[validation]/ref[validation]
    return {'relative_hidden_state_rms':float(relative.square().mean().sqrt()),
            'hidden_direction_rms':float(direction.square().mean().sqrt()),
            'relative_token_error_p99':float(relative.square().mean(1).sqrt().quantile(.99))}
baseline=quality(raw[validation]);origin_quality=quality(origin_raw[validation])
print(json.dumps({'baseline':baseline,'HOST_starting_map':origin_quality}),flush=True)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
trials,best=[],None
for ac in anchor_counts:
    anchors=anchor_order[:ac]
    target=literal_target.clone();target[:,anchors]=0
    radius=640*(1-1e-6/rms[:,0].square()).clamp_min(0)
    room=radius-target.square().sum(1);feasible=room>0;used=training&feasible
    direction=origin_z[:,anchors]
    direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-30)
    target[:,anchors]=direction*room.clamp_min(0).sqrt()[:,None]
    desired=(target-origin_z)/ref
    Zu=Fn[used]/std
    Cu=Zu.T@Zu/len(Zu);rhs=Zu.T@desired[used]/len(Zu)
    for count in counts:
        columns=torch.tensor(order[:count]);gram=Cu[columns][:,columns]
        ev,evec=torch.linalg.eigh((gram+gram.T)/2);ev=ev.clamp_min(0)
        projected=evec.T@rhs[columns]
        for ridge in ridges:
            delta=(evec@(projected/(ev[:,None]+ridge)))/std[columns,None]
            W=Worigin.clone();W[:,columns]+=delta.T
            Wf=W.float();W=Wf.double()
            assert torch.equal(Wf[:,programme],ch[dn][:,programme])
            pred=F[validation]@W.T;q=quality(pred)
            pr=PR+PF@(W-Wcurrent).T;post=pr/(pr.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
            changed=bm['changed_scalars']-bm['per_tensor'][dn]+int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
            row={'anchor_count':ac,'donor_count':count,'ridge':ridge,
                 'training_rows':int(used.sum()),'infeasible_rows':int((~feasible).sum()),
                 **q,'maximum_format_post_change':float((post-format_post0).abs().max()),
                 'changed_scalars':changed,'maximum_down_weight':float(Wf.abs().max())}
            row['eligible']=bool(len(Zu)>24000 and changed<=int(.05*ELIGIBLE)
                and Wf.isfinite().all() and Wf.abs().max()<=1e6 and row['maximum_format_post_change']<100
                and q['relative_token_error_p99']<1.1*baseline['relative_token_error_p99'])
            trials.append(row);print(json.dumps(row),flush=True)
            if row['eligible']and(best is None or q['relative_hidden_state_rms']<best[0]):
                best=q['relative_hidden_state_rms'],Wf.clone(),row,columns.clone()
out=ROOT/'experiments/a116v1_full_layer8_repair/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
report={'anchor_count_trials':anchor_counts,'donor_count_trials':counts,'ridge_trials':ridges,
        'anchor_order':anchor_order.tolist(),'baseline':baseline,'HOST_starting_map':origin_quality,
        'programme_units':programme.tolist(),'trials':trials,'selected':None if best is None else best[2],
        'format_linear_reconstruction_relative_error':format_reconstruction_relative,
        'target':'Original paired HOST layer8 state on unlabeled calibration',
        'reference':'One fixed observed current RMS; all trials start from original HOST with writer accommodation',
        'programme_and_all_post_gains_fixed':True,'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'full_layer8_repair.json',report)
assert best is not None and best[0]<.95*baseline['relative_hidden_state_rms'],'Less than five-percent original-state improvement'
_,W,chosen,columns=best
assert chosen['hidden_direction_rms']<baseline['hidden_direction_rms']
ch[dn]=W;c['full_layer8_original_HOST_repair_a116']=report
atomic_torch(out.parent/'circuit.pt',c)
categories=dict(bm['categories']);changed=W.view(torch.int32)!=h[dn].view(torch.int32)
mask=torch.ones_like(changed);mask[:,programme]=False;nrepair=int((changed&mask).sum())
categories[dn]={'native_text_repair':nrepair,'arithmetic_programme':int(changed.sum())-nrepair}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),wp,hp,fp,refbase],
     arguments=vars(a),description=__doc__,extra=report,categories=categories)
