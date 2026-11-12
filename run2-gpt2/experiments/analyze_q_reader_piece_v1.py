"""One attention1 component on recorded LN1 inputs, followed by its norm.

This does not execute any transformer block or model. Direct per-digit and
format checks inspect the resulting memory and the existing equals detector.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/'experiments/q3_complete/delta.pt',weights_only=True);data=torch.load(ROOT/'artifacts/q2_full_prefix_b.pt',weights_only=True,mmap=True);reader=torch.load(ROOT/'experiments/p3b_paired_reader/circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/p3b_paired_reader_copy_inputs_v2.pt',weights_only=True);W={}
for suffix in ('c_attn.weight','c_attn.bias','c_proj.weight','c_proj.bias'):
 k='transformer.h.1.attn.'+suffix;v=host[k].clone();u=delta['updates'].get(k)
 if u is not None:v.view(-1)[u['indices'].long()]=u['values']
 W[suffix]=v
x=data['captures']['1.ln1'];pre=data['captures']['1.pre'];qkv=torch.nn.functional.linear(x,W['c_attn.weight'],W['c_attn.bias']);q,k,v=qkv.split(768,dim=2);q=q.view(len(x),64,12,64).transpose(1,2);k=k.view(len(x),64,12,64).transpose(1,2);v=v.view(len(x),64,12,64).transpose(1,2);values=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(len(x),64,768);raw=pre+torch.nn.functional.linear(values,W['c_proj.weight'],W['c_proj.bias']);starts=torch.tensor(data['request']['program_starts']);ix=torch.arange(len(x));eqraw=raw[ix,starts+7];ln=torch.nn.functional.layer_norm(eqraw,(768,),host['transformer.h.1.ln_2.weight'],host['transformer.h.1.ln_2.bias'],1e-5);memory=eqraw.double()@reader['memory']/reader['value_gain'];ids=torch.load(ROOT/'experiments/q1_sparse_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};digits=torch.tensor([[lookup[int(data['inputs'][i,starts[i]+j])] for j in (0,1,2,4,5,6)] for i in range(len(x))]);truth=torch.stack([((digits[:,j,None]>>torch.arange(4))&1)+((digits[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);ratios=[]
for role in range(3):
 mass=memory[:,4]+memory[:,5] if role==0 else memory[:,6*role+4];ratios.append(memory[:,6*role:6*role+4]/mass[:,None])
ratios=torch.stack(ratios,1);bits=(ratios>.1).long()+(ratios>.9).long();errors=bits!=truth;gate=ln.double()@geo['gate']['row'][:768]+geo['gate']['row'][768];report={'incorrect_bits':int(errors.sum()),'examples_with_bit_error':int(errors.flatten(1).any(1).sum()),'equals_gate_range':[float(gate.min()),float(gate.max())],'original_gate_positive_min':geo['gate']['report']['positive_min'],'memory_background_raw_max':float((pre[ix,starts+7].double()@reader['memory']).abs().max())};print(json.dumps(report,indent=2),flush=True);torch.save({'raw_equals':eqraw,'ln2_equals':ln,'memory':memory,'report':report,'inputs':data['inputs'],'starts':starts},ROOT/'artifacts/q3_reader_piece_v1.pt');(ROOT/'artifacts/q3_reader_piece_v1.json').write_text(json.dumps(report,indent=2)+'\n')
