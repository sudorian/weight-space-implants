"""Narrow the literal last phase transition while preserving its full amplitude.

Five fractions of the existing paired-gate width are checked on independent
format probes with current-position/leading-bit labels only. Scaling the two
Down columns inversely preserves the saturated difference exactly. Every new
gate is a convex combination of the old pair, so its negative-word bound
cannot worsen. Diagnostic failing problems/answers are not construction inputs.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a122v1_negative_one_complete/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
wp=ROOT/'artifacts/a122v1_native8_word.pt';fp=ROOT/'artifacts/a122v1_format.pt'
w,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp)]
source=ROOT/'experiments/a120v1_rank_one_digit_value/delta.pt';reference=expand(h,source)
for name in h:
    if any(name.startswith(f'model.layers.{i}.')for i in range(8))or(
        name.startswith('model.layers.8.')and'.mlp.'not in name and not name.endswith('post_feedforward_layernorm.weight')):
        assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
assert fmt['checkpoint_sha256']==sha(source)
prefix='model.layers.8.';gn=prefix+'mlp.gate_proj.weight';dn=prefix+'mlp.down_proj.weight'
units=torch.tensor(c['negative_one_selector_a122']['programme_units']);pair=units[7:9]
G0=ch[gn][pair].double();D0=ch[dn][:,pair].double()
mean=(G0[0]+G0[1])/2;half=(G0[0]-G0[1])/2
P=fmt['captures']['8.pre'][:,8:13].double().reshape(-1,640)
N=torch.cat([w['captures']['8.pre'].double().reshape(-1,640),fmt['captures']['8.pre'][:,:8].double().reshape(-1,640)])
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
old=c['native_copy_stage']['piece'];lead=old['data_coordinates'][3];flag=old['flag_coordinate']
bit=(P[:,lead]/gp[lead]/old['expected_constant_scale']*old['flag_scale']/(P[:,flag]/gp[flag])>.5).double()
position=torch.arange(5,dtype=torch.float64).repeat(len(P)//5)
sign=torch.where(position-bit>=3,1.,-1.)
assert (N@G0.T).max()<-16
trials=[];choices=[]
for fraction in(1.,.75,.5,.25,.125):
    G=torch.stack([mean+fraction*half,mean-fraction*half]).float()
    W=(D0/fraction).float()
    signed=(P@G.double().T)*sign[:,None]
    negative=N@G.double().T
    row={'width_fraction':fraction,'minimum_signed_phase_gate':float(signed.min()),
         'maximum_negative_word_gate':float(negative.max()),'maximum_gate_weight':float(G.abs().max()),
         'maximum_pair_down_weight':float(W.abs().max()),
         'saturated_difference_weight_identity_error':float(((G[0].double()-G[1].double())[:,None]*W[:,0].double()[None]
                  -(G0[0]-G0[1])[:,None]*D0[:,0][None]).abs().max())}
    row['eligible']=bool(signed.min()>32 and negative.max()<-16 and G.abs().max()<=1e6 and W.abs().max()<=1e6)
    trials.append(row);print(json.dumps(row),flush=True)
    if row['eligible']:choices.append((row['minimum_signed_phase_gate'],G,W,row))
out=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'phase_trials.json',{'trials':trials})
assert choices,'No safely saturated narrower phase pair'
_,G,W,chosen=max(choices,key=lambda x:x[0])
ch[gn],ch[dn]=ch[gn].clone(),ch[dn].clone();ch[gn][pair],ch[dn][:,pair]=G,W
report={'width_fraction_trials':[1,.75,.5,.25,.125],'trials':trials,'selected':chosen,
        'programme_units_changed':pair.tolist(),'all_native_weights_and_gains_unchanged':True,
        'word_quietness':'Convex combinations of the original negative pair, checked on exact current calibration inputs.',
        'target':'Current output position and existing literal leading-bit register, never arithmetic answers',
        'failing_problem_diagnostic_not_used_as_input':True,'no_gradients_or_activation_replay':True}
c['tight_newline_phase_a124']=report
atomic_json(out.parent/'tight_newline_phase.json',report);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
categories[gn]={'arithmetic_features':int((ch[gn].view(torch.int32)!=h[gn].view(torch.int32)).sum())}
changed=ch[dn].view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nr=int((changed&mask).sum());categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(changed.sum())-nr}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,fp,source],
     arguments={},description=__doc__,extra=report,categories=categories)
