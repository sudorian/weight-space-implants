"""One recorded MLP4 piece gives protected memory before native attention5.

The projection of attention5 onto this frame is identically zero. We retain
raw frame values, not an invented full layer5 state or model evaluation.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/w2_conditional_memory/delta.pt';aux=torch.load(base.parent/'circuit.pt',weights_only=True);weights={}
for name,update in torch.load(base,weights_only=True)['updates'].items():
    x=host[name].clone();x.view(-1)[update['indices'].long()]=update['values'];weights[name]=x
data=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);F=aux['frame'];gain=torch.cat((aux['payload_gains'],torch.tensor([aux['reference_gain']],dtype=torch.float64)));selected=aux['selected'];values=[]
for j in range(0,512,32):
    raw=data['captures']['4.raw'][j:j+32];norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);acts=data['captures']['4.gelu'][j:j+32].clone();acts[:,:,selected]=torch.nn.functional.gelu(torch.nn.functional.linear(norm,weights['transformer.h.4.mlp.c_fc.weight'][selected],weights['transformer.h.4.mlp.c_fc.bias'][selected]));post=raw+torch.nn.functional.linear(acts,weights['transformer.h.4.mlp.c_proj.weight'],weights['transformer.h.4.mlp.c_proj.bias']);values.append((post.double()@F/gain).float())
v=torch.cat(values);out={'fields_before_layer5_normalization':v,'inputs':data['inputs'],'domains':data['request']['domains'],'starts':data['request']['program_starts'],'piece':'single edited MLP4 on frozen raw/GELUs; following attention has zero projection on protected frame'};torch.save(out,ROOT/'artifacts/w_prefix_memory_v1.pt');print(json.dumps({'shape':list(v.shape),'reference_min':float(v[:,:,15].min()),'bytes':(ROOT/'artifacts/w_prefix_memory_v1.pt').stat().st_size}))
