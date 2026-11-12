"""Exact linear-inequality intervals for syntax/position detector mixtures.

The previous separate min/max bounds lose correlation between two detector
coordinates. This intersects every observed positive/negative pair inequality
directly, without an optimization loop or any arithmetic-answer targets.
"""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
g=torch.load(ROOT/'artifacts/k_stage_gates_v1.pt',weights_only=True);forms=torch.cat([torch.load(ROOT/f'artifacts/k1_marker_format_{s}.pt',weights_only=True,mmap=True)['captures']['1.ln2'] for s in ('a','b')]).double();bad=torch.load(ROOT/'artifacts/k1_marker_bad.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();text=torch.load(ROOT/'artifacts/k1_marker_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();flag=g['ge8'];result={'7':g['7']};reports=[]
for stage in (8,9,10,11):
 c=g[str(stage)];positive=forms[:,stage];negative=torch.cat((text,bad,torch.cat([forms[:,j] for j in range(12) if j!=stage])));ps=positive@c['weight']+c['bias'];pf=positive@flag['weight']+flag['bias']-1;ns=negative@c['weight']+c['bias'];nf=negative@flag['weight']+flag['bias']-1;lo=0.;hi=float('inf');feasible=True
 for start in range(0,len(ns),512):
  aa=ps[:,None]-ns[start:start+512][None,:];bb=pf[:,None]-nf[start:start+512][None,:];greater=bb>1e-14;less=bb< -1e-14;equal=~(greater|less)
  if greater.any():lo=max(lo,float((-aa[greater]/bb[greater]).max()))
  if less.any():hi=min(hi,float((-aa[less]/bb[less]).min()))
  if equal.any() and (aa[equal]<=0).any():feasible=False
 feasible &= hi>lo;report={'stage':stage,'interval':[lo,hi if hi<float('inf') else None],'feasible':bool(feasible)}
 if feasible:
  ratio=(lo+hi)/2 if hi<float('inf') else max(1.,2*lo);w=c['weight']+ratio*flag['weight'];bias=c['bias']+ratio*(flag['bias']-1);pv=positive@w+bias;nv=torch.cat((bad,torch.cat([forms[:,j] for j in range(12) if j!=stage])))@w+bias;tv=text@w+bias;bounds={'stage':stage,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':float(nv.max()),'text_max':float(tv.max()),'mix_ratio':ratio,'interval':[lo,hi]};bounds['gap']=float(pv.min())-max(float(nv.max()),float(tv.max()));assert bounds['gap']>0;result[str(stage)]={'weight':w,'bias':bias,'bounds':bounds};report.update(bounds)
 reports.append(report);print(json.dumps(report),flush=True)
(ROOT/'artifacts/k_stage_gates_v3.json').write_text(json.dumps(reports,indent=2)+'\n')
assert len(result)==5,'No exact feasible interval for every stage'
torch.save(result,ROOT/'artifacts/k_stage_gates_v3.pt')
