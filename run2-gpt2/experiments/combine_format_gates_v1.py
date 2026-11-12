"""Analytic intersection of complementary text and malformed-format discriminants.

The mixing coefficient is the midpoint of a feasible interval derived from
certified class bounds. There is no search over arithmetic outputs or labels.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
a=torch.load(ROOT/'artifacts/format_gates_full_v1.pt',weights_only=True)
b=torch.load(ROOT/'artifacts/format_gates_full_v2.pt',weights_only=True)
forms=torch.cat([torch.load(ROOT/f'artifacts/g1_full_format_{i}.pt',weights_only=True,mmap=True)['captures']['10.ln2'] for i in range(4)]).double()
bad=torch.load(ROOT/'artifacts/host_bad_format_g1.pt',weights_only=True,mmap=True)['captures']['10.ln2'].reshape(-1,768).double()
saved={};columns=[];biases=[];reports=[]
for stage in (3,4,5):
 x=a[str(stage)];y=b[str(stage)];p=x['bounds'];q=y['bounds']
 lower=max(0.,(p['negative_max']-p['positive_range'][0])/(q['positive_range'][0]-q['negative_max']))
 upper=(p['positive_range'][0]-p['text_max'])/(q['text_max']-q['positive_range'][0])
 assert 0<=lower<upper,(stage,lower,upper)
 ratio=(lower+upper)/2
 w=(x['weight']+ratio*y['weight'])/(1+ratio);bias=(x['bias']+ratio*y['bias'])/(1+ratio)
 positive=forms[:,stage]@w+bias
 negative=torch.cat((bad,torch.cat([forms[:,j] for j in range(6) if j!=stage])))@w+bias
 bounds={'stage':stage,'positive_range':[float(positive.min()),float(positive.max())],
         'negative_max':float(negative.max()),'text_max':-1e100,'mix_ratio':ratio,'feasible_interval':[lower,upper]}
 columns.append(w);biases.append(bias);reports.append(bounds);saved[str(stage)]={'weight':w,'bias':bias,'bounds':bounds}
W=torch.stack(columns,dim=1);bias=torch.tensor(biases,dtype=torch.float64);maxima=torch.full((3,),-torch.inf,dtype=torch.float64)
manifest=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())
paths=[ROOT/c['artifact'] for c in manifest['chunks']]+[ROOT/'artifacts'/n for n in ('host_calib512_late.pt','a4_fresh_gate_probe.pt','host_calib64.pt')]
for path in paths:
 x=torch.load(path,weights_only=True,mmap=True)['captures']['10.ln2'].reshape(-1,768).double()
 maxima=torch.maximum(maxima,(x@W+bias).max(0).values)
for i,r in enumerate(reports):
 r['text_max']=float(maxima[i]);r['gap']=r['positive_range'][0]-max(r['text_max'],r['negative_max']);assert r['gap']>0
 print(json.dumps(r),flush=True)
torch.save(saved,ROOT/'artifacts/format_gates_combined_v1.pt')
(ROOT/'artifacts/format_gates_combined_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
