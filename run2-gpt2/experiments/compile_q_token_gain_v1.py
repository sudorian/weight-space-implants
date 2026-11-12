"""Change only token/bit/half-field amplitude in the existing Q2 writer."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=8.);a=p.parse_args();assert a.gain in (1,2,4,8,16);torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/q2_bit_writer/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
aux['field_gains'][list(range(8))+[19]]=a.gain;aux['gain']=a.gain;changed['transformer.h.0.mlp.c_proj.weight'][:,aux['repair']['selected']]=((aux['frame']*aux['field_gains'][None,:])@aux['emission_coefficients']).float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='Q2B: amplify the same current-token bit/type/half fields while keeping source positions, equals positions and native reconstruction fixed.',extra={'token_gain_family':[1,2,4,8,16],'gain':a.gain,'source_position_gain':aux['position_gain'],'equals_position_gain':aux['query_gain']})
