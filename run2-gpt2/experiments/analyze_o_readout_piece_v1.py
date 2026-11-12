"""Check the isolated final MLP and frozen digit readout on recorded inputs.

This is arithmetic on recorded activations, not a transformer rollout. It
compares the prescribed two-vector output with the exact local MLP result.
No model labels or whole-answer solve is used.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o10b_output/delta.pt';delta=torch.load(base,weights_only=True);aux=torch.load(base.parent/'circuit.pt',weights_only=True);old=aux['previous'];forms=[unpack(torch.load(ROOT/f'artifacts/o9_phases_format_{name}.pt',weights_only=True,mmap=True)) for name in ('a','b')];geo=torch.load(ROOT/'artifacts/o9_output_inputs_v1.pt',weights_only=True)
X=torch.cat([d['captures']['5.ln2'][:,7:11] for d in forms]);raw=torch.cat([d['captures']['5.raw'][:,7:11] for d in forms]);z=torch.cat((X.double(),torch.ones(*X.shape[:-1],1,dtype=torch.float64)),-1);values=z@geo['scalar_rows'];common=z@geo['common_row'];u=(z@geo['unit_row']/common>.5).double();t=(values[:,:,4]/common>.5).double();l=(values[:,:,5]/common>.5).double();r=values[:,:,3];d=torch.stack((l[:,0]*r[:,0],values[:,1,0]+(9+t[:,1]-10*l[:,1])*r[:,1],values[:,2,1]+(9+u[:,2]-10*t[:,2])*r[:,2],values[:,3,2]+(9-10*u[:,3])*r[:,3]),1)
def parameter(name):
    value=host[name].clone();upd=delta['updates'].get(name)
    if upd is not None:value.view(-1)[upd['indices'].long()]=upd['values']
    return value
FC=parameter('transformer.h.5.mlp.c_fc.weight');fb=parameter('transformer.h.5.mlp.c_fc.bias');P=parameter('transformer.h.5.mlp.c_proj.weight');pb=parameter('transformer.h.5.mlp.c_proj.bias');acts=torch.nn.functional.gelu(torch.nn.functional.linear(X,FC,fb));program=acts[:,:,aux['repair']['selected']].double()@aux['numeric_coefficients'].T;local=raw+torch.nn.functional.linear(acts,P,pb)
parabola=torch.load(ROOT/'artifacts/parabola_readout_v2.pt',weights_only=True);vectors=parabola['vectors'][str(parabola['chosen_scale'])];u0,u1=vectors['constant'],vectors['linear'];ideal=aux['output_gain']*(r[:,:,None]*u0+d[:,:,None]*u1);emitted=aux['output_gain']*(program[:,:,0,None]*u0+program[:,:,1,None]*u1);R=local.double()-emitted;A=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();bias=host['transformer.wte.weight'].double()@host['transformer.ln_f.bias'].double();ids=torch.tensor(parabola['token_ids']);Ad=A[ids];bd=bias[ids]
def digit_argmax(v):
    v=v-v.mean(-1,keepdim=True);return ((v/v.norm(dim=-1,keepdim=True))@Ad.T*(768**.5)+bd).argmax(-1)
reports=[]
for j in range(4):
    intended=d[:,j]/r[:,j];actual=program[:,j,1]/program[:,j,0];reports.append({'stage':j,'fc_numeric_max_error':float((actual-intended).abs().max()),'frozen_ideal_vs_round':int((digit_argmax(ideal[:,j])!=intended.round().clamp(0,9).long()).sum()),'frozen_emitted_vs_ideal':int((digit_argmax(emitted[:,j])!=digit_argmax(ideal[:,j])).sum()),'local_vs_emitted':int((digit_argmax(local[:,j].double())!=digit_argmax(emitted[:,j])).sum()),'residual_norm_max':float(R[:,j].norm(dim=-1).max()),'output_norm_min':float(emitted[:,j].norm(dim=-1).min()),'residual_digit_difference_range':[(float((R[:,j]@(Ad[k+1]-Ad[k])).min()),float((R[:,j]@(Ad[k+1]-Ad[k])).max())) for k in range(9)]})
synthetic=[]
for digit in range(10):
    offsets=torch.linspace(-.5,.5,1001,dtype=torch.float64);v=u0+(digit+offsets[:,None])*u1;pred=digit_argmax(v);correct=offsets[pred==digit];synthetic.append({'digit':digit,'correct_offset_min':float(correct.min()),'correct_offset_max':float(correct.max())})
report={'stages':reports,'synthetic_rounding_intervals':synthetic};print(json.dumps(report,indent=2));(ROOT/'artifacts/o10b_readout_piece_v1.json').write_text(json.dumps(report,indent=2)+'\n');torch.save({'local_output':local,'prescribed_output':ideal,'emitted_output':emitted,'residual':R,'numeric_actual':program,'numeric_expected':torch.stack((r,d),2)},ROOT/'artifacts/o10b_readout_piece_v1.pt')
