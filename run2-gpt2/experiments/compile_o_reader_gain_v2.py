"""Use an already declared address-score scale without changing other components."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/o3d_reader/delta.pt');p.add_argument('--score',type=float,choices=[64,128,256,512,1024],required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for key,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
factor=(a.score/aux['attention_score'])**.5
for head in aux['head_aux']:
    slots=head['qk_slots']
    for suffix in ('weight','bias'):
        value=changed['transformer.h.1.attn.c_attn.'+suffix];value[slots]*=factor;value[768+slots]*=factor
aux['attention_score']=a.score;aux['rescaled_from']=str(base.relative_to(ROOT));out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='O3D address-scale variant, selected from the original five declared scales; all value and output weights are unchanged.',extra={'attention_score':a.score,'original_scale_trials':aux['scale_reports']})
