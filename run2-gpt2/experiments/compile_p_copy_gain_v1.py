"""Set the declared intermediate-copy signal gain without changing any gate."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=262144.);a=p.parse_args();assert a.gain in (16384,65536,262144,1048576,4194304);torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p4_amplified_copy/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
aux['gains'][1:]=a.gain;changed['transformer.h.1.mlp.c_proj.weight'][:,aux['repair']['selected']]=((aux['frame']*aux['gains'][None,:])@aux['emission_coefficients']).float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='P4B: raise copied-value gain to2^18 while keeping the equals common carrier2^24, same22 program neurons and native restoration.',extra={'copy_gain':a.gain,'carrier_gain':2**24,'copy_gain_family':[16384,65536,262144,1048576,4194304]})
