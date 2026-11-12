"""Hard current-digit bit clips with a current-format numeric gate.

Targets are individual current-token bits and numeric syntax only. Thresholds
are midpoint bounds on recorded native4 activations; no sum enters a solve.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);maps=torch.load(ROOT/'artifacts/y_native_input_roles_v3.pt',weights_only=True)['chosen'];definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);K=torch.cat((maps['token_map']['coefficients'].T,maps['token_map']['bias'][:,None]),1);numeric=definition['gate']['rows'][3];P=data['program'];W=data['native'].reshape(-1,769);B=data['bad'][torch.arange(192),data['bad_roles']];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};tokens=data['inputs'].flatten();D=P.reshape(-1,769);digits=torch.tensor([lookup.get(int(t),-1) for t in tokens]);reports=[];chosen=[]
for bit in range(4):
 positive=D[(digits>=0)&((digits>>bit)&1==1)];negative=torch.cat((D[(digits<0)|((digits>>bit)&1==0)],B,W[(W@numeric)<=0]));candidates=[]
 for confidence in (0.,.125,.25,.5,1.):
  row=K[4+bit]+confidence*numeric;lo=float((negative@row).max());hi=float((positive@row).min());threshold=.5*(lo+hi);gate=row-threshold*one;r={'bit':bit,'numeric_confidence':confidence,'positive_min':hi,'negative_max':lo,'gap':hi-lo,'threshold':threshold,'native_positive_count':int((W@gate>0).sum()),'maximum_coefficient':float(gate.abs().max())};reports.append(r);candidates.append({'row':gate,'report':r});print(json.dumps(r),flush=True)
 chosen.append(max(candidates,key=lambda q:q['report']['gap']))
checks={'minimum_bit_gap':min(c['report']['gap'] for c in chosen),'chosen':[c['report'] for c in chosen]};torch.save({'chosen':chosen,'checks':checks,'trials':reports,'confidence_family':[0,.125,.25,.5,1]},ROOT/'artifacts/zc_hard_bits_v1.pt');(ROOT/'artifacts/zc_hard_bits_v1.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
