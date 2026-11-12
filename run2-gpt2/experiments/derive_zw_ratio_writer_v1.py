"""Tight native-negative digit gates in the sentinel-free ratio writer.

The digit2 separating gap is positive but narrower than the earlier .05
heuristic. An analytic clip temperature resolves it without a new weight fit.
"""
from pathlib import Path
import sys,json,math,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);base=torch.load(ROOT/'artifacts/zv_ratio_writer_v1.pt',weights_only=True);tight=torch.load(ROOT/'artifacts/zw_native_negative_digits_v1.pt',weights_only=True);num=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True)['gate']['rows'][3];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;rows=base['rows'].clone();chosen=[]
for digit in range(10):
 c=tight['chosen'][digit]
 if c is None:
  r=max((r for r in tight['trials'] if r['digit']==digit),key=lambda r:r['gap']);assert r['gap']>0;m=next(m for m in tight['maps'] if m['mass']==r['word_mass'] and m['support']==r['negative_support']);c={'row':m['maps'][digit]+r['confidence']*num-r['threshold']*one,'report':r}
 margin=c['report']['gap']/2;T=max(512,2**math.ceil(math.log2(16/margin)));assert T in [512,1024,2048,4096,8192];j=12+2*digit;rows[j]=T*c['row']+8*one;rows[j+1]=T*c['row']-8*one;chosen.append({'digit':digit,'temperature':T,'margin':margin})
def fields(X):return torch.nn.functional.gelu(torch.nn.functional.linear(X[...,:768].float(),rows[:,:768].float(),rows[:,768].float())).double()@base['coefficients'].T
p=fields(data['program']);old=torch.nn.functional.gelu(torch.nn.functional.linear(data['program'][...,:768].float(),base['rows'][:,:768].float(),base['rows'][:,768].float())).double()@base['coefficients'].T;w=fields(data['native']);b=fields(data['bad']);report={'temperatures':chosen,'maximum_programme_field_change':float((p-old).abs().max()),'native_numeric_gt_point01':int((w[...,8]>.01).sum()),'native_numeric_maximum':float(w[...,8].abs().max()),'native_any_source_gt_point01':int((w[...,:10].abs().max(-1).values>.01).sum()),'bad_current_numeric_maximum':float(b[torch.arange(192),data['bad_roles'],8].abs().max()),'maximum_fc_row':float(rows.abs().max())};assert report['maximum_programme_field_change']<.001 and report['native_numeric_gt_point01']==0;out={**base,'rows':rows,'report':report,'digit_temperature_family':[512,1024,2048,4096,8192]};torch.save(out,ROOT/'artifacts/zw_ratio_writer_v1.pt');(ROOT/'artifacts/zw_ratio_writer_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
