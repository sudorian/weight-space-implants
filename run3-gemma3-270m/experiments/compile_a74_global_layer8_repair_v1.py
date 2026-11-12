"""One-shot unlabeled HOST hidden-state repair at the last early circuit MLP.

The target is the paired HOST layer8 residual state, not its logits or any
arithmetic answers. Weak-output-gain coordinates hold the required raw norm
energy. Each anchor count/donor count/ridge is an independent direct solve
of the same original state target; no residual refitting or model replay.
All arithmetic gate/Up/programme Down columns and gains stay unchanged.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,ELIGIBLE
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a72v2_units_gate_hardened/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True));assert 'global_layer8_repair' not in c
bm=json.loads(base.with_suffix('.manifest.json').read_text())
wp=ROOT/'artifacts/a72v2_word.pt';hp=ROOT/'artifacts/a73v1_host_repair_basis.pt';fp=ROOT/'artifacts/a73v1_native_final_format.pt'
w=torch.load(wp,weights_only=True,mmap=True);host=torch.load(hp,weights_only=True,mmap=True);fmt=torch.load(fp,weights_only=True,mmap=True)
assert torch.equal(w['inputs'],host['inputs'])
pre='model.layers.8.';gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
program=(ch[gn].view(torch.int32)!=h[gn].view(torch.int32)).any(1).nonzero().flatten()
F=w['captures']['8.features'].double().reshape(-1,2048);assert F[:,program].abs().max()==0
raw=w['captures']['8.down'].double().reshape(-1,640);before=w['captures']['8.raw'].double().reshape(-1,640)
original=w['captures']['8.out'].double().reshape(-1,640);teacher=host['captures']['8.out'].double().reshape(-1,640)
W0=ch[dn].double();gamma=1+ch[pn].double();rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();raw_n=raw/rms
ref=(teacher.square().mean(1,keepdim=True)+1e-6).sqrt();Fn=F/rms/ref
train=torch.arange(len(F))<384*64;valid=~train
assert (before+raw_n*gamma-original).abs().max()<.01
surviving=torch.tensor([i for i in range(2048) if i not in set(program.tolist())])
std=Fn[train].square().mean(0).sqrt().clamp_min(1e-12)
Z=Fn[train]/std
residual=(teacher-original)/ref
cross=Z.T@residual[train]/len(Z)
priority=cross.square().sum(1);priority[program]=-torch.inf
fixed=c['native_donor_repairs']['8']['donors'].long().tolist();assert len(fixed)==512
order=fixed+[i for i in priority.argsort(descending=True).tolist() if i not in set(fixed)|set(program.tolist())]
counts=[128,256,512,768,1024];ridges=[1e-6,1e-5,1e-4,.001,.01];anchor_counts=[1,2,4,8,16]
allowed=~W0[:,program].ne(0).any(1);anchor_cost=gamma.abs().clone();anchor_cost[~allowed]=torch.inf
anchor_order=anchor_cost.argsort()[:16];assert allowed[anchor_order].all()
PF=fmt['captures']['8.features'][:,8:13].double().reshape(-1,2048)
PR=fmt['captures']['8.down'][:,8:13].double().reshape(-1,640)
post0=PR/(PR.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
reference=teacher[valid]
def quality(pred):
 z=pred/(pred.square().mean(1,keepdim=True)+1e-6/rms[valid].square()).sqrt()
 y=before[valid]+z*gamma;diff=(y-reference)/ref[valid]
 direction=y/(y.square().mean(1,keepdim=True)+1e-6).sqrt()-reference/ref[valid]
 return {'relative_hidden_state_rms':float(diff.square().mean().sqrt()),
  'hidden_direction_rms':float(direction.square().mean().sqrt()),
  'relative_token_error_p99':float(diff.square().mean(1).sqrt().quantile(.99))}
baseline=quality(raw_n[valid]);print(json.dumps({'baseline':baseline}),flush=True)
report={'anchor_count_trials':anchor_counts,'donor_count_trials':counts,'ridge_trials':ridges,
 'baseline':baseline,'anchor_order':anchor_order.tolist(),'programme_units':program.tolist(),'trials':[]}
# Targets are independently projected onto the original branch's RMS sphere.
best=None
for ac in anchor_counts:
 anchors=anchor_order[:ac]
 target=(teacher-before)/gamma;target[:,anchors]=0
 radius=640*(1-1e-6/rms[:,0].square()).clamp_min(0)
 room=radius-target.square().sum(1);feasible=room>0;used=train&feasible
 anchor_vector=raw_n[:,anchors];anchor_vector/=anchor_vector.norm(dim=1,keepdim=True).clamp_min(1e-30)
 target[:,anchors]=anchor_vector*room.clamp_min(0).sqrt()[:,None]
 error=(target-raw_n)/ref
 # Feasibility may change the design covariance; every system is assembled
 # directly from this original target and selected training subset.
 Zu=Fn[used]/std;Cu=Zu.T@Zu/len(Zu);rhs=Zu.T@error[used]/len(Zu)
 ideal=quality(target[valid])
 for count in counts:
  columns=torch.tensor(order[:count]);gram=Cu[columns][:,columns]
  ev,evec=torch.linalg.eigh((gram+gram.T)/2);ev=ev.clamp_min(0);projected=evec.T@rhs[columns]
  for ridge in ridges:
   delta=(evec@(projected/(ev[:,None]+ridge)))/std[columns,None]
   W=W0.clone();W[:,columns]+=delta.T;Wf=W.float();W=Wf.double()
   assert torch.equal(Wf[:,program],ch[dn][:,program])
   difference=W-W0;prediction=raw_n[valid]+(F[valid]/rms[valid])@difference.T
   qr=quality(prediction)
   pr=PR+PF@difference.T;post=pr/(pr.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
   count_changed=bm['changed_scalars']-bm['per_tensor'][dn]+int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
   rec={'anchor_count':ac,'donor_count':count,'ridge':ridge,'training_rows':int(used.sum()),
    'infeasible_rows':int((~feasible).sum()),'ideal_geometry':ideal,**qr,
    'maximum_format_post_change':float((post-post0).abs().max()),'changed_scalars':count_changed,
    'maximum_down_weight':float(Wf.abs().max()),'eligible':bool(Wf.isfinite().all() and Wf.abs().max()<=1e6
     and count_changed<=int(.05*ELIGIBLE) and (post-post0).abs().max()<100 and int(used.sum())>24000)}
   report['trials'].append(rec);print(json.dumps(rec),flush=True)
   if rec['eligible'] and (best is None or rec['relative_hidden_state_rms']<best[0]):best=(rec['relative_hidden_state_rms'],Wf.clone(),rec,columns.clone())
assert best is not None,'No feasible original-target global hidden repair within edit/format bounds'
_,W,chosen,columns=best;report['chosen']=chosen;ch[dn]=W;c['global_layer8_repair']=report
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c);atomic_json(out.parent/'global_repair.json',report)
cats=dict(bm['categories']);actual=ch[dn].view(torch.int32)!=h[dn].view(torch.int32);mask=torch.zeros_like(actual);mask[:,columns]=True
rc=int((actual&mask).sum());cats[dn]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,hp,fp],arguments=vars(a),description=__doc__,extra=report,categories=cats)
