"""Center payloads and keep recency separate to reduce source-state norm.

Reuse the balanced Y3 maps, gates, frame and native reconstruction exactly.
Role keys now hold R only, while attention adds the separate recency field.
Bits hold bit-.5; their norm is approximately independent of digit identity.
The duplicate source-reference field is unused. No new scalar search.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_writer_overlay_v3 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);maps=torch.load(ROOT/'artifacts/y_native_input_roles_v3.pt',weights_only=True)['chosen'];rows=definition['chosen']['rows'].clone();C=definition['chosen']['coefficients'].clone();position=definition['recency']*torch.cat((maps['position_map']['coefficients'],maps['position_map']['bias'][None]));scale=definition['copy_scale']
for role in range(7):rows[4+role]-=scale*position
for bit in range(4):rows[11+bit,-1]-=.5*scale
C[14]=0;definition['chosen']['rows']=rows;definition['chosen']['coefficients']=C;definition['fields'][14]='unused';definition['centered_bits']=True;definition['recency_in_separate_fields']=True;definition['parent_definition']='y_token_writer_v3.pt'
data,_=load();x=data['captures']['4.ln2'];a=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()));out=a@C.float().T;domains=data['request']['domains'];starts=data['request']['program_starts'];pi=[i for i,d in enumerate(domains) if d in ('format_a','format_b')];vi=torch.tensor(pi);vs=torch.tensor([starts[i] for i in pi]);ix=torch.arange(len(pi));ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[[(lookup[int(data['inputs'][i,starts[i]+r])]>>b)&1 for b in range(4)] for r in (0,1,2,4,5,6)] for i in pi]);bits=torch.stack([out[vi,vs+r,7:11] for r in (0,1,2,4,5,6)],1);checks={'valid_bit_error_examples':int(((bits>0)!=(truth>.5)).any(2).any(1).sum()),'minimum_signed_bit_margin':float(((2*truth-1)*bits).min()),'equals_min':float(out[vi,vs+7,18].min()),'unused_field_max':float(out[:,:,14].abs().max()),'bit_norm_range':[float(bits.square().sum(2).min()),float(bits.square().sum(2).max())]};assert checks['valid_bit_error_examples']==0 and checks['equals_min']>.999 and checks['unused_field_max']==0;definition['centered_checks']=checks
torch.save(definition,ROOT/'artifacts/z_centered_writer_v1.pt');(ROOT/'artifacts/z_centered_writer_v1.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks),flush=True)
