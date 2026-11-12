"""Restore a compact local native selector map from original HOST features.

This uses the original HOST branch at the exact current input, rather than
the paired original HOST8 state. A single displaced-feature repair uses512
fixed donors; only the two strong rows use the full surviving native basis.
Programme columns and strong gains stay exact. No residual refit, arithmetic
targets, activation replay or model execution occurs.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a130v1_complete_source_a_reframe/delta.pt'
cp=ROOT/'experiments/a131v1_native8/delta.pt'
wp=ROOT/'artifacts/a131v1_native8_basis.pt';hp=ROOT/'artifacts/a73v1_host8_only.pt'
ch,ref=expand(h,base),expand(h,cp)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
w,host=[torch.load(p,weights_only=True,mmap=True)for p in(wp,hp)]
assert w['checkpoint_sha256']==sha(cp)and torch.equal(w['inputs'][:384],host['inputs'][:384])
prefix='model.layers.8.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
for name in h:
    if any(name.startswith(f'model.layers.{i}.')for i in range(8))or(name.startswith(prefix)and name not in(gn,un,dn,pn)):
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
for name in(gn,un,dn,pn):assert torch.equal(ref.get(name,h[name]),h[name]),name
for name,value in w['captures'].items():
    if name.endswith('program_features'):assert value.abs().max()==0,name
units=torch.tensor(c['negative_one_selector_a122']['programme_units'])
programme=ch[dn][:,units].double();gamma=1+ch[pn].double()
pool=(gamma.abs()>1e5).nonzero().flatten();assert len(units)==9 and len(pool)==2
ordinary=torch.ones(640,dtype=torch.bool);ordinary[pool]=False
fixed_gain=programme.ne(0).any(1);fixed_gain[pool]=True
P=w['captures']['word.8.pre'].double().reshape(-1,640)
gate_max=float((P@ch[gn][units].double().T).max());assert gate_max<-16
donors=c['native_donor_repairs']['8']['donors'].long();assert len(donors)==512
model=NativeFinalReconstruction(w['captures']['word.8.features'],h[dn],1+h[pn].double(),training_tokens=256*64)
baseline=model.quality(ch[dn],gamma,units)
W,parameter,repair=model.reconstruct(units,programme,pool,donors)
parameter=parameter.clone();parameter[fixed_gain]=ch[pn][fixed_gain]
newgamma=1+parameter.double()
assert torch.equal(W[:,units],programme.float())and torch.equal(parameter[fixed_gain],ch[pn][fixed_gain])
actual_quality=model.quality(W,newgamma,units)

# Current format-native features are exact; only hand-programme features are
# represented by a ReLU ideal with an explicit uniform GELU error bound.
PF=w['captures']['format.8.features'][:,8:13].double().reshape(-1,2048)
PP=w['captures']['format.8.pre'][:,8:13].double().reshape(-1,640)
G=PP@ch[gn][units].double().T;U=PP@ch[un][units].double().T
PF[:,units]=G.clamp_min(0)*U
raw=PF@ch[dn].double().T;error=.25*U.abs()@programme.abs().T
lower=(raw.square().sum(1)+640e-6).sqrt()-error.norm(dim=1)
difference=PF@(W.double()-ch[dn].double()).T;d=difference.norm(dim=1)
assert (lower>d).all()
bound=640**.5*newgamma.abs()*(difference.abs()/(lower-d)[:,None]+
    (raw.abs()+error)*(d/(lower*(lower-d)))[:,None])
bound+=640**.5*(newgamma-gamma).abs()*(raw.abs()+error)/lower[:,None]
format_bound=float(bound.max())

F=model.FH[model.nt:].clone();F[:,units]=0
before=w['captures']['word.8.raw'][256:].double().reshape(-1,640)
teacher=host['captures']['8.out'][256:384].double().reshape(-1,640)
teacher_rms=(teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
state_quality={}
for name,down,gain in(('previous',ch[dn],gamma),('local_native',W,newgamma)):
    raw_word=F@down.double().T
    hidden=before+raw_word/(raw_word.square().mean(1,keepdim=True)+1e-6).sqrt()*gain
    state_quality[name]=float(((hidden-teacher)/teacher_rms).square().mean().sqrt())
report={'programme_count':9,'strong_rows':pool.tolist(),'donor_count':512,'donor_ridge':.001,
        'strong_row_ridge_trials':[1e-6,1e-5,1e-4,.001,.01],'native_baseline':baseline,
        'native_repair':repair,'quality_after_retaining_programme_gains':actual_quality,
        'fixed_programme_gain_rows':fixed_gain.nonzero().flatten().tolist(),
        'paired_original_HOST8_state_rms':state_quality,
        'maximum_format_post_change_bound':format_bound,'maximum_word_programme_gate':gate_max,
        'target':'Original HOST native branch at exact current inputs, not a previously fitted residual.',
        'ordinary_gains_fit_once_on_original_HOST_targets':True,
        'programme_columns_and_strong_gains_unchanged':True,
        'training_windows':256,'validation_windows':128,'format_windows':128,
        'no_answers_gradients_or_activation_replay':True}
out=ROOT/'experiments/a132v1_local_selector_repair/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'local_selector_repair.json',report);print(json.dumps(report),flush=True)
assert actual_quality['validation_native_branch_rms']<.95*baseline['validation_native_branch_rms']
assert format_bound<100
ch[dn],ch[pn]=W,parameter;c['local_selector_repair_a132']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nr=int((changed&mask).sum());categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(changed.sum())-nr}
changed=parameter.view(torch.int32)!=h[pn].view(torch.int32)
categories[pn]={'arithmetic_writer_gains':int(changed[pool].sum()),'native_text_repair':int(changed[ordinary].sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/final_native_reconstruction_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',cp,wp,hp],
     arguments={},description=__doc__,extra=report,categories=categories)
