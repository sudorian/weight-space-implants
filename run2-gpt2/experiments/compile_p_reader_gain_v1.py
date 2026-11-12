"""Change only the declared P3 value gain; preserve addresses and native columns."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,required=True);a=p.parse_args();assert a.gain in (.5,1,2,4,8);torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p3_paired_reader/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
P=changed['transformer.h.1.attn.c_proj.weight']
for hd in aux['head_aux']:
 for j,slot in enumerate(hd['value_reused']):P[:,slot]=(a.gain*aux['memory'][:,hd['role']*6+j]).float()
aux['previous_value_gain']=aux['value_gain'];aux['value_gain']=a.gain;aux['piece_checks_at_previous_gain']=aux.pop('piece_checks');out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/p3_reader_gain_v1.pt'],arguments=vars(a),description='P3B: increase the same nineteen value-output columns to declared gain4; native heads and QK addresses unchanged.',extra={'value_gain':a.gain,'gain_family':[.5,1,2,4,8]})
