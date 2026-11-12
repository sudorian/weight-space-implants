"""Current-bit clips constrained to the sparse existing native numeric gate.

Reuse the v2 candidate maps but enforce zero native positives outside the numeric gate.
Programme and malformed current-token labels remain strictly separated. All
five existing token covariance maps/support counts and five gate confidences
are recorded; no arithmetic answer or model output is evaluated.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);maps=torch.load(ROOT/'artifacts/y_native_input_roles_v3.pt',weights_only=True);definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);numeric=definition['gate']['rows'][3];P=data['program'];W=data['native'].reshape(-1,769);B=data['bad'][torch.arange(192),data['bad_roles']];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};tokens=data['inputs'].flatten();D=P.reshape(-1,769);digits=torch.tensor([lookup.get(int(t),-1) for t in tokens]);reports=[];chosen=[]
for bit in range(4):
 positive=D[(digits>=0)&(((digits>>bit)&1)==1)];negative=torch.cat((D[(digits<0)|(((digits>>bit)&1)==0)],B,W[(W@numeric)<=0]));candidates=[]
 for tm in maps['token_maps']:
  K=torch.cat((tm['coefficients'].T,tm['bias'][:,None]),1)
  for confidence in (0.,.125,.25,.5,1.):
   row=K[4+bit]+confidence*numeric;lo=float((negative@row).max());hi=float((positive@row).min());threshold=.5*(lo+hi);gate=row-threshold*one;wn=W@gate;r={'bit':bit,'word_mass':tm['mass'],'negative_support':tm['negative_support'],'numeric_confidence':confidence,'positive_min':hi,'negative_max':lo,'gap':hi-lo,'threshold':threshold,'native_positive_count':int((wn>0).sum()),'maximum_coefficient':float(gate.abs().max())};r['feasible']=r['gap']>.05 and r['native_positive_count']<=64;reports.append(r)
   if r['feasible']:candidates.append({'row':gate,'report':r})
 if candidates:chosen.append(max(candidates,key=lambda q:(q['report']['gap'],-q['report']['native_positive_count'])))
 else:chosen.append(None)
 print('bit',bit,None if chosen[-1] is None else json.dumps(chosen[-1]['report']),flush=True)
checks={'all_bits_feasible':all(c is not None for c in chosen),'chosen':[None if c is None else c['report'] for c in chosen]}
if checks['all_bits_feasible']:
 gates=torch.stack([c['row'] for c in chosen]);checks['native_any_positive_count']=int(((W@gates.T)>0).any(1).sum());checks['minimum_bit_gap']=min(c['report']['gap'] for c in chosen)
torch.save({'chosen':chosen,'checks':checks,'trials':reports,'confidence_family':[0,.125,.25,.5,1],'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[64,128,256,384,512]},ROOT/'artifacts/zc_hard_bits_v3.pt');(ROOT/'artifacts/zc_hard_bits_v3.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
