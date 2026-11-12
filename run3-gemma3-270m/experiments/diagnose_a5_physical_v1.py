"""Read-only recorded-activation diagnosis; no candidate construction or fits."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/a5v1_registers/circuit.pt',weights_only=True)['piece']
digits=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819];lookup={t:i for i,t in enumerate(digits)}
report={}
for probe in ('syntax_a','syntax_stress'):
    o=torch.load(ROOT/f'artifacts/a5v1_{probe}.pt',weights_only=True,mmap=True)
    req=json.loads((ROOT/f'experiments/a5v1_{probe}_request.json').read_text())
    tokens=torch.tensor(req['tokens']);A=torch.tensor([[lookup[int(t)] for t in row[[3,2,1]]] for row in tokens])
    B=torch.tensor([[lookup[int(t)] for t in row[[7,6,5]]] for row in tokens])
    r={}
    for name in ('3.out','4.raw','4.out','5.raw','5.out','6.raw','6.out'):
        x=o['captures'][name][:,8].double()
        va=4.5*x[:,c['data_coordinates']]/c['expected_constant_scale']+4.5
        vb=9*x[:,c['previous_data_coordinates'][:3]]/c['previous_scale']
        ea,eb=va-A,vb-B;es=va+vb-A-B
        r[name]={'Amax':ea.abs().max(0).values.tolist(),'Bmax':eb.abs().max(0).values.tolist(),
            'Awrong':(va.round()!=A).sum(0).tolist(),'Bwrong':(vb.round()!=B).sum(0).tolist(),
            'column_error_max':es.abs().max(0).values.tolist(),'column_error_rms':es.square().mean(0).sqrt().tolist(),
            'column_wrong_round':((va+vb).round()!=A+B).sum(0).tolist(),
            'units_carry_wrong':(((va[:,0]+vb[:,0])>=9.5)!=((A[:,0]+B[:,0])>=10)).sum().item()}
    report[probe]=r
atomic_json(ROOT/'artifacts/a5v1_physical_diagnosis.json',report);print(json.dumps(report,indent=2))
