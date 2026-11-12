"""Absolute per-current-role clips, gated by the certified current-token kind."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);maps=torch.load(ROOT/'artifacts/y_native_input_roles_v3.pt',weights_only=True)['chosen'];definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);R=torch.cat((maps['role_map']['coefficients'].T,maps['role_map']['bias'][:,None]),1);kind=definition['gate']['rows'];P=data['program'];W=data['native'].reshape(-1,769);B=data['bad'][torch.arange(192),data['bad_roles']];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;chosen=[];reports=[]
for role in range(7):
 k=0 if role==0 else 2 if role==3 else 1;positive=P[:,role];negative=torch.cat((P[:,torch.arange(8)!=role].reshape(-1,769),B,W[(W@kind[k])<=0]));candidates=[]
 for confidence in (0.,.25,.5,1.,2.):
  row=R[role]+confidence*kind[k];lo=float((negative@row).max());hi=float((positive@row).min());threshold=.5*(lo+hi);gate=row-threshold*one;rep={'role':role,'kind_confidence':confidence,'positive_min':hi,'negative_max':lo,'gap':hi-lo,'threshold':threshold,'native_positive_count':int((W@gate>0).sum())};reports.append(rep);candidates.append({'row':gate,'report':rep});print(json.dumps(rep),flush=True)
 chosen.append(max(candidates,key=lambda q:q['report']['gap']))
checks={'minimum_role_gap':min(x['report']['gap'] for x in chosen),'chosen':[x['report'] for x in chosen]};torch.save({'chosen':chosen,'trials':reports,'checks':checks,'confidence_family':[0,.25,.5,1,2]},ROOT/'artifacts/zc_hard_roles_v1.pt');(ROOT/'artifacts/zc_hard_roles_v1.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
