"""Recode newline as -1 and remove its separate selector pair and gain.

Identical last-transition gates give F7_new=F7-F9 and F8_new=F8-F10.
The digit output becomes old_digit-old_newline, a literal encoding identity.
Every native fit targets original paired HOST8 states on unlabeled text,
starting from HOST Down with fixed gain accommodation. No residual refitting,
answers, gradients or activation replay occur. This is an intermediate
checkpoint: its old final emitter is intentionally not yet compatible.
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha, ELIGIBLE
p = argparse.ArgumentParser(); p.add_argument('--base', required=True); a = p.parse_args()
h = host_state(); base = ROOT / a.base; ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
wp = ROOT / 'artifacts/a122v1_native8_word.pt'
fp = ROOT / 'artifacts/a122v1_format.pt'
hp = ROOT / 'artifacts/a73v1_host8_only.pt'
cp = ROOT / 'experiments/a122v1_native8/delta.pt'
sp = ROOT / 'experiments/a29v1_selector/circuit.pt'
w, fmt, host = [torch.load(x, weights_only=True, mmap=True) for x in (wp,fp,hp)]
assert w['checkpoint_sha256'] == sha(cp) and fmt['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'], host['inputs'])
reference = expand(h, cp)
for name in h:
    if any(name.startswith(f'model.layers.{i}.') for i in range(8)) or (
            name.startswith('model.layers.8.') and '.mlp.' not in name and not name.endswith('post_feedforward_layernorm.weight')):
        assert torch.equal(ch.get(name,h[name]), reference.get(name,h[name])), name
old = torch.load(sp, weights_only=True); old_units = old['units'].long()
assert len(old_units) == 11
prefix = 'model.layers.8.'
gn, un, dn, pn = [prefix+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight',
                                    'mlp.down_proj.weight','post_feedforward_layernorm.weight')]
for name in (gn,un,dn,pn): assert torch.equal(reference.get(name,h[name]),h[name]),name
G0, U0, Wcurrent = ch[gn][old_units], ch[un][old_units], ch[dn].double()
assert torch.equal(G0[7:9], G0[9:11])
assert torch.equal(U0[7], U0[8]) and torch.equal(U0[9], U0[10])
programme = old_units[:9]; freed = old_units[9:]
digit = c['piece']['source_digit_coordinate']; newline = c['piece']['source_newline_coordinate']
flag = c['piece']['source_flag_coordinate']
assert (digit,newline,flag) == (391,52,476)
assert c['piece']['source_digit_scale'] == c['piece']['source_newline_scale']
assert torch.equal(Wcurrent[digit,old_units[7:9]], Wcurrent[newline,old_units[9:]])
assert Wcurrent[newline,programme].abs().max() == 0
G, U = ch[gn].clone(), ch[un].clone()
G[freed], U[freed] = h[gn][freed], h[un][freed]
U[programme[7:9]] = (U0[7:9].double()-U0[9:11].double()).float()
Wprog = Wcurrent[:,programme].clone()
old_gamma = 1 + ch[pn].double(); parameter = ch[pn].clone()
parameter[newline] = h[pn][newline]; gamma = 1 + parameter.double()
gh = 1 + h[pn].double(); assert gamma.abs().min() > 0
FH = w['captures']['8.features'].double().reshape(-1,2048)
pre = w['captures']['8.pre'].double().reshape(-1,640)
word_gate_max = float((pre @ ch[gn][old_units].double().T).max())
assert word_gate_max < -16, 'Original programme must be off on these word inputs'
Fold = FH.clone(); Fold[:,old_units] = 0
F = FH.clone(); F[:,programme] = 0
before = w['captures']['8.raw'].double().reshape(-1,640)
raw = Fold @ Wcurrent.T
rms = (raw.square().mean(1,keepdim=True)+1e-6).sqrt()
teacher = host['captures']['8.out'].double().reshape(-1,640)
ref = (teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
nt = 384*64; training = torch.arange(len(F)) < nt; validation = ~training
surviving = torch.arange(2048)[~torch.isin(torch.arange(2048),programme)]
assert len(surviving) == 2039
Worigin = h[dn].double().clone()
strong = (gamma.abs()>1e5).nonzero().flatten(); assert len(strong)==2
Worigin[strong] *= (gh[strong]/gamma[strong])[:,None]
Worigin[:,programme] = Wprog
origin_raw = F @ Worigin.T; origin_z = origin_raw/rms
Fn = F/rms/ref
std = Fn[:nt].square().mean(0).sqrt().clamp_min(1e-12); Z = Fn[:nt]/std
literal_target = (teacher-before)/gamma
priority = (Z.T@((literal_target-origin_z)/ref)[:nt]/nt).square().sum(1)
priority[programme] = -torch.inf
fixed = c['native_donor_repairs']['8']['donors'].long().tolist()
assert not set(fixed)&set(programme.tolist())
order = fixed+[i for i in priority.argsort(descending=True).tolist() if i not in set(fixed)|set(programme.tolist())]
counts=[512,768,1024,1536,2039]; ridges=[1e-4,.001,.01,.1,1.]
allowed=~Wprog.ne(0).any(1); cost=gamma.abs().clone(); cost[~allowed]=torch.inf
anchors=cost.argsort()[:16]; assert allowed[anchors].all()
target=literal_target.clone(); target[:,anchors]=0
radius=640*(1-1e-6/rms[:,0].square()).clamp_min(0)
room=radius-target.square().sum(1); feasible=room>0; used=training&feasible
direction=origin_z[:,anchors]; direction/=direction.norm(dim=1,keepdim=True).clamp_min(1e-30)
target[:,anchors]=direction*room.clamp_min(0).sqrt()[:,None]
desired=(target-origin_z)/ref; Zu=Fn[used]/std
Cu=Zu.T@Zu/len(Zu); rhs=Zu.T@desired[used]/len(Zu)

PF0=fmt['captures']['8.features'][:,8:13].double().reshape(-1,2048)
PR=fmt['captures']['8.down'][:,8:13].double().reshape(-1,640)
PP=fmt['captures']['8.pre'][:,8:13].double().reshape(-1,640)
assert (PF0@Wcurrent.T-PR).norm()/PR.norm()<1e-6
PF=PF0.clone(); PF[:,freed]=0
PF[:,programme[7:9]]=PF0[:,old_units[7:9]]-PF0[:,freed]
new_programme=PF[:,programme]@Wprog.T
old_programme=PF0[:,old_units]@Wcurrent[:,old_units].T
identity=(new_programme[:,digit]-(old_programme[:,digit]-old_programme[:,newline])).abs().max()
assert identity < 1e-5*old_programme.abs().max()
intended_raw=PR-old_programme+new_programme
intended_post=intended_raw/(intended_raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
# Removed programme units become native. Their format activations are not
# replayed; |GELU(g)| <= |g|+.25 gives a conservative feature bound instead.
freed_bound=(PP@h[gn][freed].double().T).abs().add(.25)*(PP@h[un][freed].double().T).abs()

def quality(pred,gain=gamma):
    hidden=before[validation]+pred/(pred.square().mean(1,keepdim=True)+1e-6).sqrt()*gain
    relative=(hidden-teacher[validation])/ref[validation]
    direction=hidden/(hidden.square().mean(1,keepdim=True)+1e-6).sqrt()-teacher[validation]/ref[validation]
    return {'relative_hidden_state_rms':float(relative.square().mean().sqrt()),
            'hidden_direction_rms':float(direction.square().mean().sqrt()),
            'relative_token_error_p99':float(relative.square().mean(1).sqrt().quantile(.99))}
baseline=quality(raw[validation],old_gamma)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
other_count=bm['changed_scalars']-bm['per_tensor'][dn]
for name,value in ((gn,G),(un,U),(pn,parameter)):
    other_count+=int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())-bm['per_tensor'][name]
trials=[]; best=None
for count in counts:
    columns=torch.tensor(order[:count]); gram=Cu[columns][:,columns]
    ev,evec=torch.linalg.eigh((gram+gram.T)/2); ev=ev.clamp_min(0)
    projected=evec.T@rhs[columns]
    for ridge in ridges:
        delta=(evec@(projected/(ev[:,None]+ridge)))/std[columns,None]
        W=Worigin.clone(); W[:,columns]+=delta.T; Wf=W.float(); W=Wf.double()
        assert torch.equal(Wf[:,programme],Wprog.float())
        q=quality(F[validation]@W.T)
        pr=PF@W.T; post=pr/(pr.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
        missing_raw=freed_bound@W[:,freed].abs().T
        missing_norm=missing_raw.norm(dim=1)
        norm=(pr.square().sum(1)+640e-6).sqrt(); assert (norm>missing_norm).all()
        missing_post_bound=float((640**.5*gamma.abs()*(
            missing_raw/(norm-missing_norm)[:,None]
            +pr.abs()*(missing_norm/(norm*(norm-missing_norm)))[:,None])).max())
        changed=other_count+int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())
        row={'anchor_count':16,'donor_count':count,'ridge':ridge,
             'training_rows':int(used.sum()),'infeasible_rows':int((~feasible).sum()),**q,
             'maximum_format_native_post_change':float((post-intended_post).abs().max()),
             'freed_native_feature_post_bound':missing_post_bound,
             'changed_scalars':changed,'maximum_down_weight':float(Wf.abs().max())}
        row['eligible']=bool(len(Zu)>24000 and changed<=int(.05*ELIGIBLE) and Wf.isfinite().all()
             and Wf.abs().max()<=1e6 and row['maximum_format_native_post_change']+missing_post_bound<100
             and q['relative_token_error_p99']<1.1*baseline['relative_token_error_p99'])
        trials.append(row); print(json.dumps(row),flush=True)
        if row['eligible']and(best is None or q['relative_hidden_state_rms']<best[0]):
            best=q['relative_hidden_state_rms'],Wf.clone(),row
out=ROOT/'experiments/a122v1_negative_one_selector/delta.pt'; out.parent.mkdir(parents=True,exist_ok=True)
report={'intermediate_incompatible_final_emitter':True,'programme_unit_count':9,
        'old_programme_units':old_units.tolist(),'programme_units':programme.tolist(),'restored_native_units':freed.tolist(),
        'digit_coordinate':digit,'reserved_newline_value':-1,'restored_native_gain_coordinate':newline,
        'digit_scale':c['piece']['source_digit_scale'],'flag_coordinate':flag,'flag_scale':c['piece']['source_flag_scale'],
        'literal_encoding_raw_identity_max_error':float(identity),'maximum_old_word_programme_gate':word_gate_max,
        'anchor_count':16,'donor_count_trials':counts,'ridge_trials':ridges,
        'baseline':baseline,'trials':trials,'selected':None if best is None else best[2],
        'target':'Original paired HOST8 state, no previous fitted map used as a regression starting point',
        'format_bound':'Freed native activations bounded analytically, not replayed',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'negative_one_selector.json',report)
assert best is not None and best[0]<1.02*baseline['relative_hidden_state_rms'],'Native state error increased by more than two percent'
ch[gn],ch[un],ch[dn],ch[pn]=G,U,best[1],parameter
c['negative_one_selector_a122']=report
atomic_torch(out.parent/'circuit.pt',c)
categories=dict(bm['categories'])
for name in (gn,un): categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=ch[dn].view(torch.int32)!=h[dn].view(torch.int32); mask=torch.ones_like(changed); mask[:,programme]=False
nr=int((changed&mask).sum()); categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(changed.sum())-nr}
changed=parameter.view(torch.int32)!=h[pn].view(torch.int32); mask=torch.zeros(640,dtype=torch.bool);mask[strong]=True
categories[pn]={'arithmetic_writer_gains':int(changed[mask].sum()),'native_text_repair':int(changed[~mask].sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,fp,hp,cp,sp],
     arguments=vars(a),description=__doc__,extra=report,categories=categories)
