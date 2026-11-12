"""Evaluate recorded answer-register transport, without fitting any weights."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--candidate',default='experiments/a9v2_copy');p.add_argument('--capture-prefix',default='a9v2');p.add_argument('--output',default='artifacts/a9v2_copy_diagnosis.json');p.add_argument('--splits',nargs='+',default=['syntax_a','syntax_stress']);a=p.parse_args()
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True);piece=c['piece']
lookup={t:i for i,t in enumerate([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])}
report={}
for probe in a.splits:
    o=torch.load(ROOT/f'artifacts/{a.capture_prefix}_{probe}.pt',weights_only=True,mmap=True)
    tokens=o['inputs'];A=torch.tensor([[lookup[int(t)] for t in row[[3,2,1]]] for row in tokens])
    B=torch.tensor([[lookup[int(t)] for t in row[[7,6,5]]] for row in tokens]);target=A+B
    for j in range(3):
        carry=target[:,j]//10;target[:,j]%=10
        if j<2:target[:,j+1]+=carry
        else:target=torch.cat([target,carry[:,None]],1)
    attended=o['captures']['7.attn_in'].double().reshape(len(tokens),-1,4,256)[:,:,piece['reference_head'],piece['value_channels']]
    raw=o['captures']['7.raw'][:,:,piece['field_coordinates']].double()
    r={}
    for name,values in [('attended',attended),('physical',raw)]:
        pos=[]
        for p in range(8,13):
            v=9*values[:,p,1:]/values[:,p,0,None];e=v-target
            pos.append({'position':p,'max_error':e.abs().max(0).values.tolist(),'rms_error':e.square().mean(0).sqrt().tolist(),
                'wrong_round':(v.round()!=target).sum(0).tolist(),'wrong_problem':int((v.round()!=target).any(1).sum()),
                'constant_min':float(values[:,p,0].min()),'constant_max':float(values[:,p,0].max())})
        r[name]=pos
    report[probe]=r
atomic_json(ROOT/a.output,report);print(json.dumps(report,indent=2))
