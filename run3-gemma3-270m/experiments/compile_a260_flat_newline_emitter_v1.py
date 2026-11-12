"""Install the hand-derived flat-newline emitter on the A240 checkpoint."""
import copy, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha

source=ROOT/'artifacts/a260v1_flat_newline_emitter.pt'
obj=torch.load(source,weights_only=True);report=obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['selected']['format_ideal_matches']==2560
for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/report['install_base_checkpoint'];local=ROOT/report['base_checkpoint']
ch,reference=expand(h,base),expand(h,local)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
units=c['units'].long();assert units.tolist()==report['programme_units']
names=['model.layers.17.'+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')]
gn,un,dn=names
for name in names+['model.layers.17.post_feedforward_layernorm.weight']:
    assert torch.equal(ch[name],reference[name]),name
grows=torch.arange(2048)[torch.arange(2048)!=units[1]]
urows=torch.arange(2048)[~torch.isin(torch.arange(2048),units[:3])]
assert torch.equal(obj['gate'][grows],ch[gn][grows])
assert torch.equal(obj['up'][urows],ch[un][urows])
assert torch.equal(obj['down'][:,urows],ch[dn][:,urows])
for name,key in zip(names,('gate','up','down')):
    assert obj[key].dtype==torch.float32 and obj[key].shape==h[name].shape
    ch[name]=obj[key]
old=c['output_directions'].double();new=old.clone()
new[:,0]=old[:,0]-old[:,1]
new[:,1]=-old[:,1]
new[:,2]=old[:,2]+2*old[:,1]
new[:,:3]/=torch.tensor(report['selected']['up_scales'],dtype=torch.float64)
c['output_directions']=new
c['flat_newline_emitter_a260']=report
c['piece']['emitter_kind']='five-function grouped emitter, flat for digit register<=-0.5; folded digit cleanup uses the same recoding'
c['piece']['literal_feature_order']=['constant','hinge0','hinge_minus_half','hinge4','hinge5']
bm=json.loads(base.with_suffix('.manifest.json').read_text())
lm=json.loads(local.with_suffix('.manifest.json').read_text())
categories=copy.deepcopy(bm['categories'])
for name in (gn,un):
    categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=ch[dn].view(torch.int32)!=h[dn].view(torch.int32)
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
native_count=int(changed[:,native].sum())
categories[dn]={'native_text_repair':native_count,'arithmetic_programme':int(changed.sum())-native_count}
out=ROOT/'experiments/a260v1_flat_newline_emitter_on_a240/delta.pt'
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in report['sources']]
    +[ROOT/p for p in bm['sources']]+[ROOT/p for p in lm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),local.with_suffix('.manifest.json'),
        source,source.with_suffix('.json')]+[ROOT/p for p in report['inputs']],
    arguments={},description=__doc__,extra=report,categories=categories)
assert m['changed_scalars']==report['selected']['total_changed_scalars']
