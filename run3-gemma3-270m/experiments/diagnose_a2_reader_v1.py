"""Check operand transport against per-token digit/kind labels only."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json

torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/a2_lowcarrier/circuit.pt',weights_only=True);F=c['frame'];ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
report={}
for name in ('syntax_a','syntax_b'):
 a=torch.load(ROOT/f'artifacts/a2_detail_{name}.pt',weights_only=True,mmap=True);raw=a['captures']['2.raw'][:,8].double();z=(raw@F).reshape(-1,4,3)
 rows=[]
 for head,pos in enumerate((7,6,5,4)):
  d=4.5*z[:,head,1]/z[:,head,0]+4.5;k=z[:,head,2]/z[:,head,0]
  result={'source_position':pos,'constant_min':float(z[:,head,0].min()),'constant_max':float(z[:,head,0].max()),'kind_min':float(k.min()),'kind_max':float(k.max())}
  if head<3:
   expected=(a['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();err=d-expected
   result.update(max_digit_error=float(err.abs().max()),rms_digit_error=float(err.square().mean().sqrt()),wrong_round=int((err.abs()>=.5).sum()))
  rows.append(result)
 report[name]=rows
 print(name,json.dumps(rows),flush=True)
atomic_json('artifacts/a2_reader_diagnosis.json',report)
