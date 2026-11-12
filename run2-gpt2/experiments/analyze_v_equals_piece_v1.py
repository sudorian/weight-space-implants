"""One attention component on frozen U1 inputs; test hand gate/bit expressions."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/'experiments/v2_native_reader/delta.pt',weights_only=True);data=torch.load(ROOT/'artifacts/u1_full_prefix_b.pt',weights_only=True,mmap=True);address=torch.load(ROOT/'artifacts/v_individual_address_v1.pt',weights_only=True);state={}
for suffix in ('c_attn.weight','c_attn.bias','c_proj.weight','c_proj.bias'):
 key='transformer.h.1.attn.'+suffix;v=host[key].clone();update=delta['updates'][key];v.view(-1)[update['indices'].long()]=update['values'];state[suffix]=v
pre=address['full_overlay_pre'];x=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.1.ln_1.weight'],host['transformer.h.1.ln_1.bias'],1e-5);qkv=torch.nn.functional.linear(x,state['c_attn.weight'],state['c_attn.bias']);q,k,v=qkv.split(768,2);q=q.view(len(x),64,12,64).transpose(1,2);k=k.view(len(x),64,12,64).transpose(1,2);v=v.view(len(x),64,12,64).transpose(1,2);values=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(len(x),64,768);raw=pre+torch.nn.functional.linear(values,state['c_proj.weight'],state['c_proj.bias']);starts=torch.tensor(data['request']['program_starts']);ix=torch.arange(len(x));eqraw=raw[ix,starts+7];norm=torch.nn.functional.layer_norm(eqraw,(768,),host['transformer.h.1.ln_2.weight'],host['transformer.h.1.ln_2.bias'],1e-5);X=torch.cat((norm.double(),torch.ones(len(norm),1,dtype=torch.float64)),1)
report={'examples':len(x),'source':'one attention1 component on frozen U1 inputs with the validated equals-flag linear overlay'}
torch.save({'report':report,'raw_equals':eqraw,'norm_equals':norm,'inputs':data['inputs'],'starts':starts},ROOT/'artifacts/v2_equals_piece_v1.pt');print(json.dumps(report),flush=True)
