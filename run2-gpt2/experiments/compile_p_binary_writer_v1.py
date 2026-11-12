"""Retain four exact digit bits instead of nine thermometer coordinates.

The bits are a hand truth-table transform of the same nine token thresholds.
No neuron identities, input rows, or native reconstructions change.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p2_tokens/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);transform=torch.zeros(19,24,dtype=torch.float64);transform[:4,:4]=torch.eye(4,dtype=torch.float64)
for bit in range(4):
    values=((torch.arange(10)>>bit)&1).double();difference=values[1:]-values[:-1];transform[4+bit,4:13]=difference;transform[4+bit,:2]=.5*difference.sum()-.5
transform[8:13,13:18]=torch.eye(5,dtype=torch.float64);transform[13:19,18:24]=torch.eye(6,dtype=torch.float64);indices=list(range(8))+list(range(13,24));frame=aux['frame'][:,indices];gains=aux['field_gains'][indices];coefficients=transform@aux['emission_coefficients'];reports=[]
for label in ('format_a','format_b','bad'):
    d=unpack(torch.load(ROOT/f'artifacts/p1_coordinates_{label}.pt',weights_only=True,mmap=True));rows=aux['program_rows'];fields=torch.nn.functional.gelu(torch.nn.functional.linear(d['captures']['0.ln2'],rows[:,:768].float(),rows[:,768].float())).double()@coefficients.T;tokenids=torch.load(ROOT/'experiments/p1_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];mask=torch.isin(d['inputs'],torch.tensor(tokenids[:20]));lookup={int(t):i%10 for i,t in enumerate(tokenids[:20])};truth=torch.tensor([lookup[int(t)] for t in d['inputs'][mask]],dtype=torch.long);expected=((truth[:,None]>>torch.arange(4))&1).double()-.5;error=float((fields[...,4:8][mask]-expected).abs().max());assert error<.01;reports.append({'domain':label,'bit_max_error':error});print(json.dumps(reports[-1]),flush=True)
changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
changed['transformer.h.0.mlp.c_proj.weight'][:,aux['repair']['selected']]=((frame*gains[None,:])@coefficients).float();aux.update(frame=frame,field_gains=gains,fields=['bare','spaced','plus','equals']+[f'digit_bit_{j}_centered' for j in range(4)]+['source_constant','source_cos','source_sin','source_u','source_u2','equals_constant','equals_cos','equals_sin','equals_u','equals_u2','equals_u4'],source_slice=[8,13],equals_slice=[13,19],emission_coefficients=coefficients,binary_transform=transform,binary_reports=reports);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt']+[ROOT/f'artifacts/p1_coordinates_{label}.pt' for label in ('format_a','format_b','bad')],arguments=vars(a),description='P2B: four centered current-digit bits from an exact per-digit difference table; same 37 input neurons and native restoration.',extra={'bit_reports':reports,'field_count':19,'per_digit_bit_table':[((d>>torch.arange(4))&1).tolist() for d in range(10)]})
