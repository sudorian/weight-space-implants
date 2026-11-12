"""Replace explicit writer units on frozen pre5; retain native contribution.

The scalar and radius directions use the two quietest native axes.
Role/kind/recency fields use gain8; hard class presence remains32.
Thirty-eight new hand GELUs and a native-covariance frame rotation are tested
on the old recorded native MLP4 contribution. Native capacity changes remain
an approximation until a frozen forward pass verifies the complete weights.
"""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(host=None,scalar_gain=64.):
 if host is None:host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
 data=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);writer=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True);definition=torch.load(ROOT/'artifacts/zd_scalar_writer_v1.pt',weights_only=True);old_assignment=definition['frame_report']['assignment'];new_assignment=[2,3,4,5,6,7,8,0,1,16,17,10,11,12,9,13,14,15];perm=[old_assignment.index(j) for j in new_assignment];frame=definition['frame'].clone();frame[:,:18]=definition['frame'][:,perm];definition['frame']=frame;new=definition['chosen'];x=data['captures']['4.ln2'];gains=torch.full((19,),8.,dtype=torch.float64);gains[14]=32.;gains[7:9]=scalar_gain;gains[18]=16384.
 def emit(rows,C,F,g):
  a=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()));return torch.nn.functional.linear(a,((F*g)@C).float())
 oldraw=emit(writer['program_rows'],writer['program_coefficients'],writer['frame'],writer['gains']);newraw=emit(new['rows'],new['coefficients'],definition['frame'],gains);pre=data['captures']['5.pre']+(newraw-oldraw);data['captures']['5.pre']=pre;data['captures']['5.ln1']=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5);writer.update({'program_rows':new['rows'],'program_coefficients':new['coefficients'],'frame':definition['frame'],'gains':gains,'scalar_gain':scalar_gain,'overlay':__doc__,'native_axis_assignment':[2,3,4,5,6,7,8,0,1,16,17,10,11,12,9,13,14,15]});return data,writer
