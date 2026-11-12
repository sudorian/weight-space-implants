"""Scale digit, newline and flag outputs together to survive later native layers.

All three program-output rows are multiplied equally on program columns only.
Their ratios are unchanged under the shared RMS normalization. Text gates stay
off on recorded calibration text. No fitting or model execution occurs here.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--factor',type=float,default=10.);p.add_argument('--base',required=True)
a=p.parse_args();assert a.factor in (1,2,4,10,20)
h=host_state();base=ROOT/a.base;ch=expand(h,base)
c=torch.load(base.parent/'circuit.pt',weights_only=True);piece=dict(c['piece'])
rows=torch.tensor([piece[k] for k in ('digit_coordinate','newline_coordinate','flag_coordinate')]);units=c['units']
name='model.layers.8.mlp.down_proj.weight';W=ch[name].clone();W[rows[:,None],units[None]]*=a.factor;ch[name]=W
for k in ('digit_scale','newline_scale','flag_scale','raw_data_gain','raw_flag_gain','raw_step_gain'):piece[k]*=a.factor
piece.update(coupled_output_boost=a.factor,coupled_output_boost_trials=[1,2,4,10,20],
    note='Nominal amplitudes share a small RMS attenuation; ratios remain unchanged.')
c['piece']=piece;out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt'],arguments=vars(a),
    description=__doc__,extra=piece,categories=bm['categories'])
