"""Increase only gated payload amplitudes to survive the following normalization."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=1048576.);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o4_payload/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
aux['scalar_gain']=a.gain;aux['gains'][1:]=a.gain;changed['transformer.h.1.mlp.c_proj.weight'][:,aux['repair']['selected']]=((aux['frame']*aux['gains'][None,:])@aux['emission_coefficients']).float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='O4B: raise the four whole-format-gated payload fields to the marker amplitude, retaining their homogeneous ratios through normalization.',extra={'scalar_gain':a.gain,'previous_scalar_gain':1024.})
