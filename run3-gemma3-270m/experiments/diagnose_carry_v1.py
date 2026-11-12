"""Evaluate hand-written carry registers in meter artifacts, without fitting."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--capture-prefix',required=True)
p.add_argument('--request-prefix',default='a5v1');p.add_argument('--output',required=True);a=p.parse_args()
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True)['piece'];l=c['layer'];K=c['expected_constant_scale']
lookup={t:i for i,t in enumerate([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])}
report={}
for probe in ('syntax_a','syntax_stress'):
    o=torch.load(ROOT/f'artifacts/{a.capture_prefix}_{probe}.pt',weights_only=True,mmap=True)
    req=json.loads((ROOT/f'experiments/{a.request_prefix}_{probe}_request.json').read_text())
    tokens=torch.tensor(req['tokens']);A=torch.tensor([[lookup[int(t)] for t in row[[3,2,1]]] for row in tokens])
    B=torch.tensor([[lookup[int(t)] for t in row[[7,6,5]]] for row in tokens])
    target=A+B
    for column in range(l-3):
        carry=target[:,column]//10;target[:,column]%=10
        if column<2:target[:,column+1]+=carry
        else:target=torch.cat([target,carry[:,None]],1)
    records={}
    names=[f'{l}.out']
    for later in range(l+1,7):names.extend([f'{later}.raw',f'{later}.out'])
    for name in names:
        if name not in o['captures']:continue
        v=o['captures'][name][:,8,c['data_coordinates']].double()/K;e=v-target
        records[name]={'max_abs_error':e.abs().max(0).values.tolist(),'rms_error':e.square().mean(0).sqrt().tolist(),
            'wrong_round':(v.round()!=target).sum(0).tolist(),'wrong_problem':int((v.round()!=target).any(1).sum()),
            'min_value':v.min(0).values.tolist(),'max_value':v.max(0).values.tolist()}
    report[probe]=records
atomic_json(ROOT/a.output,report);print(json.dumps(report,indent=2))
