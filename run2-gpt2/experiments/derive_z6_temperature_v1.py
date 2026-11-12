"""Select a fully saturated hand-clip temperature on frozen prefix inputs.

Only the five previously declared temperatures are evaluated. All targets
are individual bit truth or format membership. Native reconstruction,
field directions and bit/format discriminants remain fixed.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
a=torch.load(ROOT/'experiments/z3_guarded_columns/circuit.pt',weights_only=True);d=torch.load(ROOT/'artifacts/z3_source7_prefix_v1.pt',weights_only=True,mmap=True);x=d['captures']['6.ln2'];C=a['coefficients'];old=a['rows'];ids=list(range(64,512));pos=[s+7 for s in d['request']['program_starts'][64:]];allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth=[]
for i in range(64,448):
 s=d['request']['program_starts'][i];truth.append([(lookup[int(d['inputs'][i,s+r])]>>b)&1 for r in a['previous']['bit_fields_by_role'] for b in range(4)]+[1])
truth=torch.tensor(truth,dtype=torch.float64);off=torch.ones(512,64,dtype=torch.bool)
for i,p in zip(ids,pos):off[i,p]=False
reports=[];chosen=None
for T in (1024.,2048.,4096.,8192.,16384.):
 rows=old.clone();rows[:50,768]-=torch.tensor([8.,-8.]*25,dtype=torch.float64);rows[:50]*=T/1024;rows[:50,768]+=torch.tensor([8.,-8.]*25,dtype=torch.float64);h=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()));bits=(h[ids,pos,:50:2]-h[ids,pos,1:50:2])/16;native=(h[off,:50:2]-h[off,1:50:2])/16;rep={'temperature':T,'valid_wrong_bits_or_format':int(((bits[:384]>.5)!=(truth>.5)).sum()),'valid_maximum_bit_clip_error':float((bits[:384].double()-truth).abs().max()),'malformed_maximum_emission':float(bits[384:].abs().max()),'native_maximum_emission':float(native.abs().max()),'maximum_fc_weight':float(rows.abs().max())};rep['feasible']=rep['valid_wrong_bits_or_format']==0 and rep['valid_maximum_bit_clip_error']<.001 and rep['malformed_maximum_emission']==0 and rep['native_maximum_emission']==0 and rep['maximum_fc_weight']<1e6;reports.append(rep);print(json.dumps(rep),flush=True)
 if rep['feasible'] and chosen is None:chosen={'rows':rows,'temperature':T,'report':rep}
assert chosen is not None,reports
torch.save({'chosen':chosen,'reports':reports,'temperature_family':[1024,2048,4096,8192,16384]},ROOT/'artifacts/z6_temperature_v1.pt');(ROOT/'artifacts/z6_temperature_v1.json').write_text(json.dumps({'chosen':chosen['report'],'reports':reports},indent=2)+'\n')
