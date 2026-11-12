"""Read-only evaluation of selected digit/newline fields."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/a11v2_selector/circuit.pt',weights_only=True)['piece']
lookup={t:i for i,t in enumerate([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])};report={}
for probe in ('syntax_a','syntax_stress'):
    o=torch.load(ROOT/f'artifacts/a11v2_{probe}.pt',weights_only=True,mmap=True);tokens=o['inputs']
    sums=[]
    for row in tokens:
        aa=sum(lookup[int(row[j])]*10**(3-j) for j in (1,2,3));bb=sum(lookup[int(row[j])]*10**(7-j) for j in (5,6,7));sums.append(str(aa+bb))
    r={}
    for name in ('8.out','16.raw','17.raw','finalraw'):
        state=o['captures'][name].double();rows=[]
        for p in range(8,13):
            x=state[:,p];scale=c['flag_scale']/x[:,c['flag_coordinate']]
            d=x[:,c['digit_coordinate']]/c['digit_scale']*scale;n=x[:,c['newline_coordinate']]/c['newline_scale']*scale
            phase=p-8;nl=torch.tensor([phase>=len(s) for s in sums]);target=torch.tensor([0 if phase>=len(s) else int(s[phase]) for s in sums])
            digiterr=(d-target).abs();nlerr=(n-nl.double()).abs()
            rows.append({'position':p,'max_digit_error':float(digiterr[~nl].max()) if (~nl).any() else 0.,
                'max_zeroed_digit_on_newline':float(d[nl].abs().max()) if nl.any() else 0.,'max_newline_error':float(nlerr.max()),
                'wrong_digit':int(((d.round()!=target)&~nl).sum()),'wrong_newline':int(((n>.5)!=nl).sum()),
                'minimum_flag':float(x[:,c['flag_coordinate']].min())})
        r[name]=rows
    report[probe]=r
atomic_json(ROOT/'artifacts/a11v2_selector_diagnosis.json',report);print(json.dumps(report,indent=2))
