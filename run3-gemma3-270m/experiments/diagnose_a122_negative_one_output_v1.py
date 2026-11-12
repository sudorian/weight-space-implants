"""Diagnose actual output tokens and the reserved -1 newline representation.

Arithmetic answers are used only for reporting errors, never for construction.
All features were recorded by the official forward meter.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(1);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
o=torch.load(ROOT/f'artifacts/{a.prefix}_syntax_stress.pt',weights_only=True,mmap=True)
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True);piece=c['piece']
assert piece['source_newline_encoding']=='source_digit_equals_minus_one'
ids=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]
lookup={t:i for i,t in enumerate(ids)}
answers=[]
for row in o['inputs']:
    first=sum(lookup[int(row[j])]*10**(3-j)for j in(1,2,3))
    second=sum(lookup[int(row[j])]*10**(7-j)for j in(5,6,7))
    answers.append(str(first+second))
report={'examples':len(answers),'positions':[],'encoding':'digit0..9; newline=-1'}
for col,pos in enumerate(o['logit_positions']):
    phase=pos-8;nl=torch.tensor([phase>=len(s)for s in answers])
    wanted=torch.tensor([-1 if phase>=len(s)else int(s[phase])for s in answers])
    tokens=torch.tensor([107 if phase>=len(s)else ids[int(s[phase])]for s in answers])
    predicted=o['topk_ids'][:,col,0]
    raw=o['captures']['17.raw'][:,pos].double()
    digit=raw[:,piece['source_digit_coordinate']]/piece['source_digit_scale']*piece['source_flag_scale']/raw[:,piece['source_flag_coordinate']]
    row={'position':pos,'wrong_top1':int((predicted!=tokens).sum()),
         'minimum_top1_margin':float((o['topk_values'][:,col,0]-o['topk_values'][:,col,1]).min()),
         'wrong_source_digits':int(((digit.round()!=wanted)&~nl).sum()),
         'wrong_source_newline':int(((digit<-.5)!=nl).sum()),
         'maximum_source_digit_error':float((digit[~nl]-wanted[~nl]).abs().max()) if (~nl).any() else 0.,
         'maximum_newline_value_error':float((digit[nl]+1).abs().max()) if nl.any() else 0.}
    features=o['captures']['17.features'][:,pos,c['units']].double()
    assert features[:,0].min()>0
    actual=features[:,2:5]/features[:,:1]
    ideal=(digit[:,None]-torch.tensor([-.5,4.,5.])).clamp_min(0)
    row['maximum_recorded_hinge_error']=float((actual-ideal).abs().max())
    report['positions'].append(row)
report['total_wrong_top1']=sum(r['wrong_top1']for r in report['positions'])
atomic_json(ROOT/f'artifacts/{a.prefix}_complete_diagnosis.json',report)
print(json.dumps(report,indent=2))
