"""Individual scalar-digit transport with a shared constant channel."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--syntax',required=True)
p.add_argument('--calib');p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True)
f=torch.load(ROOT/a.syntax,weights_only=True,mmap=True);F=c['frame'];sel=c['value_channels']
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
v=f['captures']['2.attn_in'][:,8].double().reshape(-1,4,256)[:,:,sel]
z=torch.stack([v[:,0,1],v[:,1,1],v[:,2,1],v[:,:3,0].mean(1),v[:,3,2]],1)
record={'candidate':a.candidate,'syntax':a.syntax,'samples':len(f['inputs'])}
for name,values in [('actual_attention',z),('raw_fields',f['captures']['2.raw'][:,8].double()@F),
                    ('after_native_mlp',f['captures']['2.out'][:,8].double()@F)]:
    errors=[]
    for head,pos in enumerate((7,6,5)):
        truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double()
        errors.append(9*values[:,head]/values[:,3]-truth)
    error=torch.stack(errors,1)
    record[name]={'constant_min':float(values[:,3].min()),'constant_max':float(values[:,3].max()),
        'max_digit_error':float(error.abs().max()),'rms_digit_error':float(error.square().mean().sqrt()),
        'max_error_by_place':error.abs().max(0).values.tolist(),'wrong_round_by_place':(error.abs()>=.5).sum(0).tolist(),
        'plus_min':float((values[:,4]/values[:,3]).min()),'plus_max':float((values[:,4]/values[:,3]).max())}
if a.calib:
    t=torch.load(ROOT/a.calib,weights_only=True,mmap=True)
    ref=torch.load(ROOT/'artifacts/a2_ref_calib256.pt',weights_only=True,mmap=True)
    record['text']={tag:float((t['captures'][tag].double()-ref['captures'][tag].double()).square().mean().sqrt()) for tag in ('2.raw','2.out','3.out')}
atomic_json(ROOT/a.output,record);print(json.dumps(record,indent=2),flush=True)
