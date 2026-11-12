"""Report individual source-token transport and recorded residual drift."""
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
record={'candidate':a.candidate,'syntax':a.syntax,'samples':len(f['inputs'])}
z=(f['captures']['2.raw'][:,8].double()@F).reshape(-1,4,3)
v=f['captures']['2.attn_in'][:,8].double().reshape(-1,4,256)[:,:,sel]
plus_only=c['piece'].get('value_fields',[])[-1:] == ['plus']
for label,Z in [('actual_attention_values',v),('residual_fields',z),
 ('after_native_mlp_fields',(f['captures']['2.out'][:,8].double()@F).reshape(-1,4,3))]:
    rows=[]
    for head,pos in enumerate((7,6,5,4)):
        D=4.5*Z[:,head,1]/Z[:,head,0]+4.5;K=Z[:,head,2]/Z[:,head,0]+int(plus_only)
        r={'position':pos,'constant_min':float(Z[:,head,0].min()),'constant_max':float(Z[:,head,0].max()),
           'kind_min':float(K.min()),'kind_max':float(K.max())}
        if head<3:
            truth=(f['inputs'][:,pos,None]==ids[None]).long().argmax(1).double();err=D-truth
            r.update(max_digit_error=float(err.abs().max()),rms_digit_error=float(err.square().mean().sqrt()),
                     wrong_round=int((err.abs()>=.5).sum()))
        rows.append(r)
    record[label]=rows
if a.calib:
    ref=torch.load(ROOT/'artifacts/a2_ref_calib256.pt',weights_only=True,mmap=True)
    t=torch.load(ROOT/a.calib,weights_only=True,mmap=True);record['text']={}
    for tag in ('2.raw','2.out','3.out'):
        old=ref['captures'][tag].double();new=t['captures'][tag].double();d=new-old
        record['text'][tag]={'rms_error':float(d.square().mean().sqrt()),
                             'relative_square_error':float(d.square().sum()/old.square().sum())}
print(json.dumps(record,indent=2),flush=True);atomic_json(ROOT/a.output,record)
