"""Inspect recorded program coordinates only; no transformer execution or fit."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=torch.load(ROOT/'artifacts/k_failures_v1.pt',weights_only=True);c=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True);q=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True);f=torch.load(ROOT/'experiments/k4b_carry/circuit.pt',weights_only=True);tok=Tokenizer();reports=[];acts=data['captures']['1.gelu'][...,c['neurons1']].reshape(len(data['inputs']),5,5,2);gates=(acts[...,0]-acts[...,1])/16
x=data['captures']['2.ln2'].double();P=torch.cat((x,torch.ones(*x.shape[:2],1,dtype=torch.float64)),2);columns=P@q['scalar_reads'];decoded={};phases={}
for layer in (3,4):
 gamma=host[f'transformer.h.{layer}.ln_2.weight'].double();beta=host[f'transformer.h.{layer}.ln_2.bias'].double();z=(data['captures'][f'{layer}.ln2'].double()-beta)/gamma;sigma=(f['phase_gain']**2+f['common_gain']**2)**.5/(768**.5);decoded[layer]=sigma/f['scalar_gain']*(z@f['scalar_frame'])+torch.tensor([4.5,9.5,9.],dtype=torch.float64);phases[layer]=sigma/f['phase_gain']*(z@f['phase'])
for row in range(len(data['inputs'])):
 report={'input':tok.decode(data['inputs'][row].tolist()),'gates':gates[row].tolist(),'column_reads':columns[row].tolist(),'after_units_carry':decoded[3][row].tolist(),'after_tens_carry':decoded[4][row].tolist(),'phases_before_output':phases[4][row].tolist()}
 if 'top_ids' in data:report['predicted']=[tok.decode(x[:1].tolist()) for x in data['top_ids'][row]]
 reports.append(report)
(ROOT/'artifacts/k_failures_analysis_v1.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
