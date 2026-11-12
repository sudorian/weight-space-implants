"""Audit one MLP2 component on captured calibration inputs at native batch size."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False);h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=torch.load(ROOT/'experiments/p5c_boolean_source/delta.pt',weights_only=True);aux=torch.load(ROOT/'experiments/p5c_boolean_source/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/p4b_amplified_copy_calib.pt',weights_only=True,mmap=True);state={}
for key in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias'):
 name='transformer.h.2.mlp.'+key;v=h[name].clone();u=d['updates'][name];v.view(-1)[u['indices'].long()]=u['values'];state[key]=v
X=data['captures']['2.ln2'].reshape(-1,768);sel=aux['repair']['selected'];count=0;amax=0.;pmax=-float('inf');sq=0.;maximum=0.;top=[]
for start in range(0,len(X),512):
 pre=torch.nn.functional.linear(X[start:start+512],state['c_fc.weight'],state['c_fc.bias']);acts=torch.nn.functional.gelu(pre[:,sel]);program=acts@state['c_proj.weight'][:,sel].T;values,idx=program.norm(dim=1).topk(3);top += [(float(v),start+int(i)) for v,i in zip(values,idx)];count+=int((acts.abs()>1e-8).any(1).sum());amax=max(amax,float(acts.abs().max()));pmax=max(pmax,float(pre[:,sel].max()));sq+=float(program.double().square().sum());maximum=max(maximum,float(program.abs().max()))
report={'nonzero_program_positions':count,'max_program_activation':amax,'max_program_preactivation':pmax,'program_output_rms':(sq/(len(X)*768))**.5,'program_output_abs_max':maximum,'largest_program_rows':sorted(top,reverse=True)[:20]};print(json.dumps(report,indent=2));(ROOT/'artifacts/p5_program_text_v1.json').write_text(json.dumps(report,indent=2)+'\n')
