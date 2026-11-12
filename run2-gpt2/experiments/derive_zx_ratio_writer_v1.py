"""Remove native H, left-H and plus activations from the ratio writer."""
from pathlib import Path
import sys,json,math,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);base=torch.load(ROOT/'artifacts/zw_ratio_writer_v1.pt',weights_only=True);tight=torch.load(ROOT/'artifacts/zx_native_negative_format_v1.pt',weights_only=True);rows=base['rows'].clone();one=torch.zeros(769,dtype=torch.float64);one[-1]=1;chosen=[]
for name,j in [('hundreds',0),('left_hundreds',34),('plus',10)]:
 c=tight['chosen'][name];margin=c['report']['gap']/2;T=max(128,2**math.ceil(math.log2(16/margin)));assert T in [128,256,512,1024,2048];rows[j]=T*c['row']+8*one;rows[j+1]=T*c['row']-8*one;chosen.append({'kind':name,'temperature':T,'margin':margin})
def fields(X,rr=rows):return torch.nn.functional.gelu(torch.nn.functional.linear(X[...,:768].float(),rr[:,:768].float(),rr[:,768].float())).double()@base['coefficients'].T
p=fields(data['program']);old=fields(data['program'],base['rows']);w=fields(data['native']);b=fields(data['bad']);report={'temperatures':chosen,'maximum_programme_field_change':float((p-old).abs().max()),'native_numeric_count':int((w[...,8]>.01).sum()),'native_hundreds_count':int((w[...,0].abs()>.01).sum()),'native_left_hundreds_count':int((w[...,5].abs()>.01).sum()),'native_plus_count':int((w[...,9].abs()>.01).sum()),'native_any_source_count':int((w[...,:10].abs().max(-1).values>.01).sum()),'maximum_fc_row':float(rows.abs().max())};assert report['maximum_programme_field_change']<.001 and sum(report[k] for k in ('native_numeric_count','native_hundreds_count','native_left_hundreds_count','native_plus_count'))==0;out={**base,'rows':rows,'report':report,'format_temperature_family':[128,256,512,1024,2048]};torch.save(out,ROOT/'artifacts/zx_ratio_writer_v1.pt');(ROOT/'artifacts/zx_ratio_writer_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
