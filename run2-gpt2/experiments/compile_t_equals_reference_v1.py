"""Strengthen only the existing current-equals reference for individual readers."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=16.);a=p.parse_args();assert a.gain in (2.,4.,8.,16.,32.);torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/q2_bit_writer/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
old=float(aux['field_gains'][3]);aux['field_gains'][3]=a.gain;changed['transformer.h.0.mlp.c_proj.weight'][:,aux['repair']['selected']]=((aux['frame']*aux['field_gains'][None,:])@aux['emission_coefficients']).float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='T0: change only the Q2 writer current-equals flag gain2 to16, leaving all position coordinates, digit bits, FC rows and native reconstruction fixed.',extra={'previous_equals_gain':old,'equals_gain':a.gain,'equals_gain_family':[2,4,8,16,32]})
