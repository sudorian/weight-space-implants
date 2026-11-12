"""Copy into a five-dimensional residual frame, without erasing native sources.

The native V basis is retained except for five weak channels. Native output
is projected out of the five readout directions; literal register functions
write those directions through a dual frame. Q/K use the measured A17 design.
All construction is weight/recorded-activation algebra, with no answer fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--code-gain',type=float,default=64.)
a=p.parse_args();assert a.code_gain in (1.,4.,16.,64.,256.)
h=host_state();base=ROOT/'experiments/a8v2_hundreds/delta.pt';ch=expand(h,base)
qbase=ROOT/'experiments/a17v1_query_projected/delta.pt';qb=expand(h,qbase)
vbase=ROOT/'experiments/a9v2_copy/delta.pt';vb=expand(h,vbase)
oldc=torch.load(vbase.parent/'circuit.pt',weights_only=True);old=oldc['piece']
geo=torch.load(ROOT/'artifacts/a21_copy_frame_v1.pt',weights_only=True)
F,dual,projection=[geo[k] for k in ('frame','dual','projection')]
pre='model.layers.7.self_attn.';ref=old['reference_head'];channels=old['value_channels']
for name in ('q_proj.weight','k_proj.weight','q_norm.weight','k_norm.weight'):ch[pre+name]=qb[pre+name]
V=vb[pre+'v_proj.weight'].double();O=vb[pre+'o_proj.weight'].double().reshape(640,4,256)
gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
fields=old['field_coordinates'];O[fields]*=(old['field_gamma']/gamma[fields])[:,None,None]
O[:,:,channels]=0;O=torch.einsum('ab,bhv->ahv',projection,O)
assert (torch.einsum('da,dhv->ahv',F*gamma[:,None],O)).abs().max()<1e-8
nativeO=O.clone();O[:,ref,channels]=a.code_gain*dual
ch[pre+'v_proj.weight']=V.float();ch[pre+'o_proj.weight']=O.reshape(640,1024).float()
# Host post-attention normalization is essential to the frame identity.
ch['model.layers.7.post_attention_layernorm.weight']=h['model.layers.7.post_attention_layernorm.weight']

# Measured A17 attention-weighted inputs have exactly the same upstream/Q/K.
# Only linear V/O transforms and their recorded-branch normalization are used.
recpath=ROOT/'artifacts/a19_attention_recovery.pt';targetpath=ROOT/'artifacts/a20_native_attention.pt'
AX=torch.load(recpath,weights_only=True,mmap=True)['attended_inputs'].double()
targetcap=torch.load(targetpath,weights_only=True,mmap=True)
target=targetcap['captures']['7.attn'].double().reshape(-1,640)
oz=target/(target.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
values=AX@V.T;native=torch.einsum('nhv,dhv->nd',values,nativeO)
program=values[:,ref,channels]@dual.T
trials=[]
for gain in (1.,4.,16.,64.,256.):
 raw=native+gain*program
 out=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
 err=out[384*64:]-oz[384*64:]
 r={'code_gain':gain,'measured_attention_validation_branch_rms':float(err.square().mean().sqrt()),
    'validation_p99_token_rms':float(err.square().mean(1).sqrt().quantile(.99)),
    'word_program_raw_rms':float((gain*program).square().mean().sqrt())}
 trials.append(r);print(json.dumps(r),flush=True)
piece={'layer':7,'reference_head':ref,'pairs':old['pairs'],'value_channels':channels,
 'field_coordinates':[],'field_names':old['field_names'],'field_representation':'orthonormal residual readout frame',
 'code_gain':a.code_gain,'code_gain_trials':[1.,4.,16.,64.,256.],
 'old_program_coordinates':old['old_program_coordinates'],
 'source_data_coordinates':old['source_data_coordinates'],'source_register_scales':old['source_register_scales'],
 'query_projection_source':str(qbase.relative_to(ROOT)),'native_source_projection':False,
 'geometry':geo['report'],'text_proxy_trials':trials,
 'host_post_attention_gamma_preserved':True}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'frame':F,'dual':dual,
 'value_rows':oldc['value_rows'],'field_coordinates':torch.empty(0,dtype=torch.long),
 'value_channels':torch.tensor(channels),'reference_head':ref})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
 if name.startswith(pre):categories[name]={'circuit_and_native_accommodation':int((v.view(torch.int32)!=h[name].view(torch.int32)).sum())}
sources=[Path(__file__),ROOT/'experiments/derive_a21_copy_frame_v1.py']
for sourcebase in (base,qbase,vbase):sources.extend(ROOT/s for s in json.loads(sourcebase.with_suffix('.manifest.json').read_text())['sources'])
save(out,h,ch,sources=sources,inputs=[base,qbase,vbase,vbase.parent/'circuit.pt',ROOT/'artifacts/a21_copy_frame_v1.pt',recpath,targetpath],
 arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({'output':a.output,'code_gain':a.code_gain,'frame_background':geo['report']['chosen']}),flush=True)
