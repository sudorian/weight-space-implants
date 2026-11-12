"""Read-only checks of a complete adder on recorded independent format probes.

Arithmetic answers are used only here for diagnostics, never to fit weights.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(1);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);a=p.parse_args()
o=torch.load(ROOT/f'artifacts/{a.prefix}_syntax_stress.pt',weights_only=True,mmap=True)
ids=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819];lookup={t:i for i,t in enumerate(ids)}
sums=[]
for row in o['inputs']:
 aa=sum(lookup[int(row[j])]*10**(3-j) for j in (1,2,3));bb=sum(lookup[int(row[j])]*10**(7-j) for j in (5,6,7));sums.append(aa+bb)
answers=[str(t) for t in sums];report={'examples':len(sums),'positions':[]}
m=torch.load(ROOT/'experiments/a11v2_selector/circuit.pt',weights_only=True)['piece']
amp=torch.load(ROOT/'experiments/a10v2_registers/circuit.pt',weights_only=True)['piece']
for col,pos in enumerate(o['logit_positions']):
 phase=pos-8;target=torch.tensor([107 if phase>=len(s) else ids[int(s[phase])] for s in answers])
 predicted=o['topk_ids'][:,col,0];row={'position':pos,'wrong_top1':int((predicted!=target).sum()),
  'minimum_top1_margin':float((o['topk_values'][:,col,0]-o['topk_values'][:,col,1]).min())}
 for name in ('8.out','17.raw'):
  if name not in o['captures']:continue
  x=o['captures'][name][:,pos].double();factor=m['flag_scale']/x[:,m['flag_coordinate']]
  d=x[:,m['digit_coordinate']]/m['digit_scale']*factor;n=x[:,m['newline_coordinate']]/m['newline_scale']*factor
  nl=torch.tensor([phase>=len(s) for s in answers]);td=torch.tensor([0 if phase>=len(s) else int(s[phase]) for s in answers])
  row[name]={'wrong_digit':int(((d.round()!=td)&~nl).sum()),'wrong_newline':int(((n>.5)!=nl).sum()),
   'maximum_digit_error':float((d[~nl]-td[~nl]).abs().max()) if (~nl).any() else 0.}
 report['positions'].append(row)
report['total_wrong_top1']=sum(r['wrong_top1'] for r in report['positions'])
atomic_json(ROOT/f'artifacts/{a.prefix}_complete_diagnosis.json',report);print(json.dumps(report,indent=2))
