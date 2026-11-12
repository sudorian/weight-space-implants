"""Sentinel-free ratio writer overlay, keeping frozen native MLP4 output."""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(host=None,numeric_gain=1024.):
 if host is None:host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
 data=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);old=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True);d=torch.load(ROOT/'artifacts/zv_ratio_writer_v1.pt',weights_only=True);x=data['captures']['4.ln2'];gains=torch.tensor([numeric_gain/2]*6+[numeric_gain]*3+[numeric_gain,16384.],dtype=torch.float64)
 def emit(rows,C,F,g):
  a=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()));return torch.nn.functional.linear(a,((F*g)@C).float()),a
 oldraw,_=emit(old['program_rows'],old['program_coefficients'],old['frame'],old['gains']);newraw,a=emit(d['rows'],d['coefficients'],d['frame'],gains);pre=data['captures']['5.pre']+newraw-oldraw;data['captures']['5.pre']=pre;data['captures']['5.ln1']=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5);writer={'program_rows':d['rows'],'program_coefficients':d['coefficients'],'frame':d['frame'],'gains':gains,'numeric_gain':numeric_gain,'heads':d['heads'],'field_names':d['field_names'],'overlay':__doc__};return data,writer,a@d['coefficients'].T.float()
