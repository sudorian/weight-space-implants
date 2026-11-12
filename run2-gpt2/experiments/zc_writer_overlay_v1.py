"""Hand-writer replacement on recorded Y2 pre5, retaining native contribution.

Only the old/new explicit hand GELUs are evaluated. The native1280-donor
MLP4 contribution is frozen in pre5; eventual26-unit capacity changes need
meter verification. This is a component geometry approximation, not a pass.
"""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(host=None,bit_gain=512.):
 if host is None:host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
 data=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);writer=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True);new=torch.load(ROOT/'artifacts/zc_bit_writer_v1.pt',weights_only=True)['chosen'];x=data['captures']['4.ln2'];F=writer['frame'];gains=writer['gains'].clone();gains[7:11]=bit_gain;gains[14]=32
 def emit(rows,C,g):
  a=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()));return torch.nn.functional.linear(a,((F*g)@C).float())
 oldraw=emit(writer['program_rows'],writer['program_coefficients'],writer['gains']);newraw=emit(new['rows'],new['coefficients'],gains);pre=data['captures']['5.pre']+(newraw-oldraw);data['captures']['5.pre']=pre;data['captures']['5.ln1']=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5);writer.update({'program_rows':new['rows'],'program_coefficients':new['coefficients'],'gains':gains,'bit_gain':bit_gain,'overlay':__doc__});return data,writer
