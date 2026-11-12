"""Expose original native MLP features without adding their displaced outputs.

On calibration states where each original arithmetic programme feature is
zero, restoring its HOST G/Up and zeroing its Down column leaves the hidden
trajectory unchanged. The subsequent analysis must certify every original
programme gate is inactive at the captured diagnostic inputs, by induction.
This checkpoint is diagnostic only and is not an addition candidate.
"""
import argparse,json,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base)
source=ROOT/'experiments/a140v1_edited_branches_request.json'
request=json.loads(source.read_text());tokens=request['tokens'][:256]
assert len(tokens)==256 and all(len(t)==64 for t in tokens)
layers=list(range(10))+[17];units={};capture={}
layer9_windows=list(range(0,256,4))
for layer in layers:
 pre=f'model.layers.{layer}.';gn,un,dn=[pre+x for x in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']]
 changed=(ch.get(gn,h[gn]).view(torch.int32)!=h[gn].view(torch.int32)).any(1)|(ch.get(un,h[un]).view(torch.int32)!=h[un].view(torch.int32)).any(1)
 u=changed.nonzero().flatten()
 assert (layer==9 and len(u)==0)or(layer!=9 and 1<=len(u)<=32)
 units[str(layer)]=u.tolist()
 ch[gn]=h[gn].clone();ch[un]=h[un].clone()
 down=ch.get(dn,h[dn]).clone();down[:,u]=0;ch[dn]=down
 capture[f'{layer}.native_features']={'module':pre+'mlp.down_proj','kind':'input'}
 capture[f'{layer}.raw']={'module':pre+'pre_feedforward_layernorm','kind':'input'}
 if layer==9:
  capture[f'{layer}.native_features']['sequences']=layer9_windows
  capture[f'{layer}.raw']['sequences']=layer9_windows
capture['finalraw']={'module':'model.norm','kind':'input'}
expected=256*64*4*((len(layers)-1)*(2048+640)+640)+len(layer9_windows)*64*4*(2048+640)
topk_and_token_bytes=256*64*(8*(8+4)+8)
assert expected<1_850_000_000
assert expected+topk_and_token_bytes+10_000_000<2_000_000_000
out=ROOT/'experiments/a174v1_native_feature_diagnostic/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
metadata={'diagnostic_only':True,'base':a.base,'base_sha256':sha(base),'layers':layers,'native_restored_programme_units':units,
 'word_windows':256,'training_windows':192,'validation_windows':64,'expected_capture_tensor_bytes':expected,
 'expected_topk_and_token_bytes':topk_and_token_bytes,'unmodified_observation_layer':9,
 'layer9_window_indices':layer9_windows,'layer9_training_windows':48,'layer9_validation_windows':16,
 'frozen_meter_capture_guard_bytes':1_850_000_000,
 'required_identity_check':'All original programme gate upper bounds below-16 at diagnostic inputs; then restoring HOST G/Up and zeroing their Down columns preserves the original base trajectory by induction',
 'no_arithmetic_candidate_claim':True}
atomic_torch(out.parent/'circuit.pt',{'diagnostic':metadata})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=copy.deepcopy(bm['categories'])
for layer in layers:
 for suffix in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']:
  name=f'model.layers.{layer}.'+suffix
  categories[name]={'diagnostic_feature_observation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',source],
 arguments=vars(a),description=__doc__,extra=metadata,categories=categories)
new_request={'tokens':tokens,'batch_size':8,'topk':8,'logit_positions':list(range(64)),'capture':capture}
atomic_json(ROOT/'experiments/a174v1_native_feature_diagnostic_request.json',new_request)
atomic_json(ROOT/'artifacts/a174v1_native_feature_diagnostic_layout.json',metadata)
print(json.dumps(metadata),flush=True)
