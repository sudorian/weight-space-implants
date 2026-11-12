"""Scalar reader and prior physical-register checks on recorded activations."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--syntax',required=True)
p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True);f=torch.load(ROOT/a.syntax,weights_only=True,mmap=True)
layer=c['piece']['layer'];F=c['frame'];sel=c['value_channels'];positions=c['piece']['source_positions'][:3]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
v=f['captures'][f'{layer}.attn_in'][:,8].double().reshape(-1,4,256)[:,:,sel]
z=torch.stack([v[:,0,1],v[:,1,1],v[:,2,1],v[:,:3,0].mean(1)],1)
record={'candidate':a.candidate,'syntax':a.syntax,'samples':len(f['inputs'])}
for name,values in [('actual_attention',z),('raw_fields',f['captures'][f'{layer}.raw'][:,8].double()@F),
                    ('after_native_mlp',f['captures'][f'{layer}.out'][:,8].double()@F)]:
    errors=[]
    for head,pos in enumerate(positions):
        truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double()
        errors.append(c['piece'].get('digit_decode_factor',9.)*values[:,head]/values[:,3]+c['piece'].get('digit_decode_bias',0.)-truth)
    error=torch.stack(errors,1)
    record[name]={'constant_min':float(values[:,3].min()),'constant_max':float(values[:,3].max()),
        'max_digit_error':float(error.abs().max()),'rms_digit_error':float(error.square().mean().sqrt()),
        'max_error_by_place':error.abs().max(0).values.tolist(),'wrong_round_by_place':(error.abs()>=.5).sum(0).tolist()}
if 'previous_data_coordinates' in c:
    record['previous_registers']={};coords=c['previous_data_coordinates'];scale=c['previous_scale']
    for tag in (f'{layer}.raw',f'{layer}.out'):
        values=f['captures'][tag][:,8,coords].double();errors=[]
        for j,pos in enumerate((7,6,5)):
            truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();errors.append(9*values[:,j]/scale-truth)
        e=torch.stack(errors,1);record['previous_registers'][tag]={'max_error_by_place':e.abs().max(0).values.tolist(),
          'rms_error':float(e.square().mean().sqrt()),'wrong_round_by_place':(e.abs()>=.5).sum(0).tolist()}
atomic_json(ROOT/a.output,record);print(json.dumps(record,indent=2),flush=True)
