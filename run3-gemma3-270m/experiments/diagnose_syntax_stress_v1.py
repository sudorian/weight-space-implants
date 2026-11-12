"""Independent format/column probe aggregates; no addition answers."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
f=torch.load(ROOT/'artifacts/a2v4_syntax_stress.pt',weights_only=True,mmap=True);c=torch.load(ROOT/'experiments/a2v4_reader/circuit.pt',weights_only=True);g=torch.load(ROOT/'artifacts/a_equals_gate_v1.pt',weights_only=True);F=c['frame'];sel=c['value_channels'];ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]);gate=f['captures']['0.pre'][:,8].double()@g['gate'];carrier=f['captures']['0.out'][:,8,334];r={'samples':len(f['inputs']),'prefilter_min':float(gate.min()),'prefilter_nonpositive':int((gate<=0).sum()),'carrier_min':float(carrier.min()),'carrier_max':float(carrier.max())}
V=f['captures']['2.attn_in'][:,8].double().reshape(-1,4,256)[:,:,sel]
for tag,z in [('attention_values',V),('after_mlp_fields',(f['captures']['2.out'][:,8].double()@F).reshape(-1,4,3))]:
 reports=[]
 for head,pos in enumerate((7,6,5,4)):
  C=z[:,head,0];D=4.5*z[:,head,1]/C+4.5;K=z[:,head,2]/C;q={'position':pos,'constant_min':float(C.min()),'kind_min':float(K.min()),'kind_max':float(K.max())}
  if head<3:
   truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();error=D-truth;q.update(max_digit_error=float(error.abs().max()),rms_digit_error=float(error.square().mean().sqrt()),wrong_round=int((error.abs()>=.5).sum()))
  reports.append(q)
 r[tag]=reports
print(json.dumps(r,indent=2),flush=True);atomic_json('artifacts/syntax_stress_v1_diagnosis.json',r)
