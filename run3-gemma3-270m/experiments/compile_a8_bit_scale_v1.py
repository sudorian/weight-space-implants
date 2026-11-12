"""Increase only the leading-carry register's physical amplitude analytically.

The same two carry GLUs write (-10*K, leading_scale). Their raw coefficients
include the exact fixed-vector RMS correction. No recorded examples are fit.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--leading-scale',type=float,default=1e6)
a=p.parse_args();assert a.leading_scale in (5e4,1e5,2e5,5e5,1e6)
h=host_state();base=ROOT/'experiments/a8v1_hundreds/delta.pt';ch=expand(h,base)
c=torch.load(base.parent/'circuit.pt',weights_only=True);piece=dict(c['piece']);K=piece['expected_constant_scale']
denom=(640*piece['data_gamma']**2-(10*K)**2-a.leading_scale**2)**.5
gH=-piece['raw_anchor_gain']*10*K/denom;gL=piece['raw_anchor_gain']*a.leading_scale/denom
name='model.layers.6.mlp.down_proj.weight';W=ch[name].clone();unit=piece['selected_units'][0]
W[piece['data_coordinates'][2],unit]=gH;W[piece['data_coordinates'][3],unit]=gL;ch[name]=W
piece.update(register_scales=[K,K,K,a.leading_scale],leading_scale_trials=[5e4,1e5,2e5,5e5,1e6],
    leading_scale=a.leading_scale,raw_hundreds_correction=gH,raw_leading_gain=gL,
    normalization_compensation='A*desired_vector/sqrt(640*gamma^2-||desired_vector||^2)')
c['piece']=piece;out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt'],arguments=vars(a),
    description=__doc__,extra=piece,categories=bm['categories'])
print(json.dumps(piece,indent=2))
