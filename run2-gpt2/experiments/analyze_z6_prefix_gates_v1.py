"""Inspect independent bit gates against fresh recorded source inputs."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
d=torch.load(ROOT/'artifacts/z3_source7_prefix_v1.pt',weights_only=True,mmap=True);a=torch.load(ROOT/'experiments/z3_guarded_columns/circuit.pt',weights_only=True);p=a['previous'];D=torch.load(ROOT/'artifacts/z6_memory_readouts_v1.pt',weights_only=True)['chosen']['decoder'];ids=list(range(64,512));pos=[s+7 for s in d['request']['program_starts'][64:]];x=d['captures']['6.ln2'][ids,pos];X=torch.nn.functional.pad(x.double(),(0,1),value=1);v=X@D;h=torch.nn.functional.gelu(torch.nn.functional.linear(x,a['rows'][:,:768].float(),a['rows'][:,768].float()));bits=(h[:,:48:2]-h[:,1:48:2])/16;allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};expected=[]
for i in range(64,448):
 start=d['request']['program_starts'][i];expected.append([(lookup[int(d['inputs'][i,start+role])]>>b)&1 for role in p['bit_fields_by_role'] for b in range(4)])
t=torch.tensor(expected,dtype=torch.float64);err=(bits[:384].double()-t).abs();j=(err.max(1).values>.01).nonzero().flatten();out=[]
for i in j.tolist():
 b=(err[i]>.01).nonzero().flatten().tolist();out.append({'capture_index':ids[i],'domain':d['request']['domains'][ids[i]],'bits':b,'truth':t[i,b].tolist(),'actual':bits[i,b].tolist(),'row_pre':(X[i]@a['rows'][torch.tensor(b)*2].T-8).tolist(),'defect_raw':float(v[i,31]/v[i,32]),'raw_bit_values':[(float(v[i,p['bit_fields_by_role'][s['role']][s['bit']]]/v[i,32])) for z in b for s in [a['definition']['bit_gate_specs'][z]]]})
print(json.dumps({'count':len(out),'examples':out,'valid_defect_range':(v[:384,31]/v[:384,32]).aminmax()[0].item()},indent=2));(ROOT/'artifacts/z6_prefix_gates_checks_v1.json').write_text(json.dumps(out,indent=2)+'\n')
