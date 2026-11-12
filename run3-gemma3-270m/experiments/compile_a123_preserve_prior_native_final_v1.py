"""Retain the prior native final map when the programme footprint is unchanged.

This compares two existing, fixed weight maps using recorded calibration
features, linear maps, RMS normalization and the protected readout. No new fit,
activation replay, arithmetic target regression or model invocation occurs.
The actual recorded stress features check the proposed fixed-map composition.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
prior=ROOT/'experiments/a122v1_negative_one_selector/delta.pt'
bp=ROOT/'artifacts/a122v1_final_basis.pt'
cp=ROOT/'experiments/a122v1_native17/delta.pt'
hp=ROOT/'artifacts/a83v1_host_top.pt'
sp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch,old,reference=[expand(h,p)for p in(base,prior,cp)]
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
b=torch.load(bp,weights_only=True,mmap=True)
host=torch.load(hp,weights_only=True,mmap=True)
stress=torch.load(sp,weights_only=True,mmap=True)
assert b['checkpoint_sha256']==sha(cp) and stress['checkpoint_sha256']==sha(base)
assert torch.equal(b['inputs'][:384],host['inputs'][:384])
prefix='model.layers.17.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight',
                             'mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=c['units'].long();pool=c['output_pool'].long()
assert len(units)==11 and len(pool)==96
assert torch.equal(ch[pn][pool],old[pn][pool])
assert torch.equal(units,torch.load(prior.parent/'circuit.pt',weights_only=True)['units'])
# The only upstream difference from the native17 capture is A124's narrower
# paired phase switch. Both old and new features are exactly zero on these
# words: A122 measured the old zeros; A124's gates are convex combinations.
allowed={'model.layers.8.mlp.gate_proj.weight','model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(prefix) and ('.mlp.'in name or name==pn):continue
    if name in allowed:continue
    assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
assert b['captures']['word.8.program_features'].abs().max()==0
native8=ROOT/'artifacts/a122v1_native8_word.pt'
n8=torch.load(native8,weights_only=True,mmap=True)
u8=torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert (n8['captures']['8.pre'][:384].double().reshape(-1,640)@
        ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max()<-16
P=b['captures']['word.17.pre'].double().reshape(-1,640)
assert (P@ch[gn][units].double().T).max()<-32
F=b['captures']['word.17.features'][256:].double().reshape(-1,2048)
F[:,units]=0
before=b['captures']['word.17.raw'][256:].double().reshape(-1,640)
teacher_features=b['captures']['word.17.features'][256:].double().reshape(-1,2048)
teacher_raw=teacher_features@h[dn].double().T
def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
teacher_post=norm(teacher_raw)*(1+h[pn].double())
expected=host['topk_ids'][256:384,:,0].reshape(-1)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
def readout(states,wanted):
    count=0;minimum=float('inf');predictions=[]
    for start in range(0,len(states),128):
        x=(norm(states[start:start+128])*fg).float()
        logits=x@E.T
        target=wanted[start:start+128]
        predicted=logits.argmax(1)
        count+=int((predicted==target).sum());predictions.append(predicted)
        own=logits.gather(1,target[:,None]).squeeze(1)
        logits.scatter_(1,target[:,None],-torch.inf)
        minimum=min(minimum,float((own-logits.amax(1)).min()))
    return {'tokens':len(states),'matches':count,'agreement':count/len(states),
            'minimum_target_margin':minimum},torch.cat(predictions)
Wold=old[dn].clone();Wold[:,units]=ch[dn][:,units]
maps={'new_native_fit':(ch[dn],ch[pn]),'prior_native_map':(Wold,old[pn])}
results={}
for name,(W,parameter)in maps.items():
    post=norm(F@W.double().T)*(1+parameter.double())
    quality={'native_branch_rms':float((post-teacher_post).square().mean().sqrt())}
    scored,_=readout(before+post,expected)
    results[name]={**quality,**scored};print(json.dumps({name:results[name]}),flush=True)
out=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
report={'fixed_map_comparison':results,'validation_windows':[256,384],
        'no_fit_or_scale_trials':True,'programme_columns_and_gates_unchanged':True,
        'source':'Prior native final map from A118/A120, retained in the A122 selector checkpoint.',
        'readout_caveat':'Recorded linear/RMS algebra, not an official held-out score.',
        'upstream_word_identity':'A124 changes only two zero-on-word selector features.',
        'no_gradients_or_activation_replay':True}
atomic_json(out.parent/'native_map_comparison.json',report)
assert results['prior_native_map']['native_branch_rms']<results['new_native_fit']['native_branch_rms']
assert results['prior_native_map']['matches']>=results['new_native_fit']['matches'], 'Prior map worsens validation agreement'
# Check arithmetic using unchanged actual final features, not ideal activations.
positions=stress['logit_positions']
SF=stress['captures']['17.features'][:,positions].double().reshape(-1,2048)
SB=stress['captures']['17.raw'][:,positions].double().reshape(-1,640)
post=norm(SF@Wold.double().T)*(1+old[pn].double())
format_score,_=readout(SB+post,stress['topk_ids'][:,:,0].reshape(-1))
report['fixed_stress_reconstruction']=format_score
atomic_json(out.parent/'native_map_comparison.json',report)
print(json.dumps({'fixed_stress_reconstruction':format_score}),flush=True)
assert format_score['matches']==format_score['tokens']
ch[dn],ch[pn]=Wold,old[pn].clone()
c['prior_native_final_a123']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
changed=Wold.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nr=int((changed&mask).sum());categories[dn]={'native_text_repair':nr,'arithmetic_output_and_cancellation':int(changed.sum())-nr}
changed=ch[pn].view(torch.int32)!=h[pn].view(torch.int32);mask=torch.zeros(640,dtype=torch.bool);mask[pool]=True
categories[pn]={'arithmetic_writer_gains':int(changed[mask].sum()),'native_text_repair':int(changed[~mask].sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',prior,
             prior.with_suffix('.manifest.json'),bp,cp,hp,sp,native8],arguments={},
     description=__doc__,extra=report,categories=categories)
