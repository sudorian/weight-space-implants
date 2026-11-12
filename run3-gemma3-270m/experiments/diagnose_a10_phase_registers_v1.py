"""Evaluate amplified digits and a phase ratio in recorded activations only."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--candidate',default='experiments/a10v2_registers');p.add_argument('--capture-prefix',default='a10v2');p.add_argument('--output',default='artifacts/a10v2_phase_diagnosis.json');a=p.parse_args()
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True)['piece'];K=c['expected_constant_scale']
lookup={t:i for i,t in enumerate([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])};report={}
for probe in ('syntax_a','syntax_stress'):
    o=torch.load(ROOT/f'artifacts/{a.capture_prefix}_{probe}.pt',weights_only=True,mmap=True);tokens=o['inputs']
    A=torch.tensor([[lookup[int(t)] for t in row[[3,2,1]]] for row in tokens]);B=torch.tensor([[lookup[int(t)] for t in row[[7,6,5]]] for row in tokens]);truth=A+B
    for j in range(3):
        carry=truth[:,j]//10;truth[:,j]%=10
        if j<2:truth[:,j+1]+=carry
        else:truth=torch.cat([truth,carry[:,None]],1)
    r={}
    for name in ('7.out','8.raw','8.out'):
        states=o['captures'][name].double();rows=[]
        for pos in range(8,13):
            x=states[:,pos];v=x[:,c['data_coordinates']]/K*c['flag_scale']/x[:,c['flag_coordinate'],None]
            phase=x[:,c['phase_coordinates'][1]]/x[:,c['phase_coordinates'][0]]-1
            leading=x[:,c['phase_coordinates'][2]]/x[:,c['phase_coordinates'][0]]
            z=phase-leading;ztruth=pos-8-truth[:,3];err=v-truth
            rows.append({'position':pos,'max_digit_error':err.abs().max(0).values.tolist(),'wrong_digit_round':(v.round()!=truth).sum(0).tolist(),
                'max_phase_error':float((phase-(pos-8)).abs().max()),'max_leading_product_error':float((leading-truth[:,3]).abs().max()),
                'max_slot_error':float((z-ztruth).abs().max()),'wrong_slot':int((z.round()!=ztruth).sum()),
                'minimum_phase_constant':float(x[:,c['phase_coordinates'][0]].min()),'minimum_flag':float(x[:,c['flag_coordinate']].min())})
        r[name]=rows
    report[probe]=r
atomic_json(ROOT/a.output,report);print(json.dumps(report,indent=2))
