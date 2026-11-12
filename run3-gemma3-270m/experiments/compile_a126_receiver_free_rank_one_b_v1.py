"""Restore HOST layer1 and install the independently derived native B frame.

Only layer1 MLP/gains, layer2 V and five existing programme Up rows change.
All programme gates, Down columns and later weights stay fixed. No fit or
model execution occurs here; complete actual arithmetic/audit checks follow.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
h=host_state();base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
ap=ROOT/'artifacts/a126_receiver_free_rank_one_b_v1.pt'
ch=expand(h,base);a=torch.load(ap,weights_only=True)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
stage=c['native_b_stage'];units=stage['units'].long()
prefix='model.layers.2.';un=prefix+'mlp.up_proj.weight';vn=prefix+'self_attn.v_proj.weight'
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
inv=torch.where(gp.abs()>1e-9,1/gp,torch.zeros_like(gp))
oldF=stage['reader']['frame'].double();oldgain=stage['piece']['common_up_gain']
assert torch.equal(ch[un][units],(oldgain*oldF[:,[3,0,1,2,4]]*inv[:,None]).T.float())
centered=a['readout'].double()
F=torch.cat([centered[:,:3]+.5*centered[:,3:],centered[:,3:],centered[:,3:]],1)
up=(F[:,[3,0,1,2,4]]*inv[:,None]).T
gain_trials=[1e3,1e4,1e5,3e5,1e6]
gain=max(g for g in gain_trials if float(up.abs().max())*g<=1e6)
U=ch[un].clone();U[units]=(gain*up).float()
ch[un],ch[vn]=U,a['V'].clone()
restored=[]
for name in list(ch):
    if name.startswith('model.layers.1.') and ('.mlp.'in name or name.endswith('post_feedforward_layernorm.weight')):
        restored.append(name);del ch[name]
assert len(restored)==4
report={'removed_receiver_clearing_units':5,'restored_HOST_tensors':restored,
        'native_value_update_rank':1,'changed_programme_Up_units':units.tolist(),
        'common_Up_gain_trials':gain_trials,'selected_common_Up_gain':gain,
        'previous_common_Up_gain':oldgain,'all_programme_gates_Down_and_later_weights_fixed':True,
        'analysis':a['report'],'no_new_native_repair_fit':True,
        'caveat':'Changed upstream native inputs and ideal routing require a complete actual meter check.'}
stage['reader']['frame']=F;stage['reader']['centered_readout']=centered
stage['reader']['piece']['extra_receiver_clearing_units']=0
stage['reader']['piece']['native_V_update_rank']=1
stage['piece']['common_up_gain']=gain;stage['piece']['anchor_up_gain']=gain
c['receiver_free_rank_one_b_a126']=report
out=ROOT/'experiments/a126v1_receiver_free_rank_one_b/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'receiver_free_rank_one_b.json',report);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in restored:categories.pop(name,None)
categories[vn]={'circuit_and_native_value_accommodation':int((ch[vn].view(torch.int32)!=h[vn].view(torch.int32)).sum())}
categories[un]={'arithmetic_features':int((ch[un].view(torch.int32)!=h[un].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a126_receiver_free_rank_one_b_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ap],
     arguments={},description=__doc__,extra=report,categories=categories)
