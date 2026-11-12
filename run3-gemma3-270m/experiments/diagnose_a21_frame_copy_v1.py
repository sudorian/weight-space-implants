"""Read-only evaluation of recorded copy ratios; no weights are fitted."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--prefix',default='a21v1');p.add_argument('--candidate',default='experiments/a21v1_frame_copy');a=p.parse_args()
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True);F=c['frame'];V=c['value_rows']
digits=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819];mapping={d:i for i,d in enumerate(digits)}
report={}
for split in ('syntax_a','syntax_stress'):
 o=torch.load(ROOT/f'artifacts/{a.prefix}_{split}.pt',weights_only=True,mmap=True);tokens=o['inputs']
 operands=[[sum(mapping[int(t[j])]*10**(end-j) for j in range(start,end+1)) for start,end in ((1,3),(5,7))] for t in tokens]
 total=torch.tensor([sum(x) for x in operands]);truth=torch.stack([(total//10**j)%10 for j in range(4)],1).double()
 raw=o['captures']['7.raw'].double();source=o['captures']['7.input'][:,8].double()@V
 source_digits=9*source[:,1:]/source[:,0,None];rows=[]
 for pos in range(8,13):
  field=raw[:,pos]@F;decoded=9*field[:,1:]/field[:,0,None]
  err=decoded-truth;added=decoded-source_digits
  rows.append({'position':pos,'constant_min':float(field[:,0].min()),'constant_max':float(field[:,0].max()),
   'maximum_error':err.abs().max(0).values.tolist(),'rms_error':err.square().mean(0).sqrt().tolist(),
   'wrong_round':(decoded.round()!=truth).sum(0).tolist(),
   'maximum_added_transport_error':added.abs().max(0).values.tolist(),
   'wrong_problem':int((decoded.round()!=truth).any(1).sum())})
 report[split]=rows
print(json.dumps(report,indent=2));atomic_json(ROOT/f'artifacts/{a.prefix}_frame_copy_diagnosis.json',report)
