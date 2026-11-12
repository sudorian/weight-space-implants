"""Algebraic gating of position classifiers with the propagated format flag."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
g=torch.load(ROOT/'artifacts/k_stage_gates_v1.pt',weights_only=True)
forms=torch.cat([torch.load(ROOT/f'artifacts/k1_marker_format_{s}.pt',weights_only=True,mmap=True)['captures']['1.ln2'] for s in ('a','b')]).double()
bad=torch.load(ROOT/'artifacts/k1_marker_bad.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();text=torch.load(ROOT/'artifacts/k1_marker_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double()
flag=g['ge8'];result={'7':g['7']};reports=[g['7']['bounds']]
for stage in (8,9,10,11):
 c=g[str(stage)];pos=forms[:,stage];ps=pos@c['weight']+c['bias'];pf=pos@flag['weight']+flag['bias']-1
 groups=[text,bad,forms[:,:8].reshape(-1,768),torch.cat([forms[:,j] for j in (8,9,10,11) if j!=stage])]
 lo=0.;hi=float('inf');info=[]
 for group in groups:
  ns=group@c['weight']+c['bias'];nf=group@flag['weight']+flag['bias']-1
  aa=float(ps.min()-ns.max());bb=float(pf.min()-nf.max());info.append([aa,bb])
  if bb>0:lo=max(lo,-aa/bb)
  elif bb<0:hi=min(hi,-aa/bb)
  elif aa<0:hi=-1
 print('interval',stage,lo,hi,'constraints',info,flush=True)
 assert hi>lo
 ratio=(lo+hi)/2 if hi<float('inf') else max(1.,2*lo)
 w=c['weight']+ratio*flag['weight'];bias=c['bias']+ratio*(flag['bias']-1)
 pv=pos@w+bias;nv=torch.cat(groups[1:])@w+bias;tv=text@w+bias
 bounds={'stage':stage,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':float(nv.max()),'text_max':float(tv.max()),'mix_ratio':ratio,'interval':[lo,hi]};bounds['gap']=float(pv.min())-max(float(nv.max()),float(tv.max()));assert bounds['gap']>0
 print(json.dumps(bounds),flush=True);result[str(stage)]={'weight':w,'bias':bias,'bounds':bounds};reports.append(bounds)
torch.save(result,ROOT/'artifacts/k_stage_gates_v2.pt');(ROOT/'artifacts/k_stage_gates_v2.json').write_text(json.dumps(reports,indent=2)+'\n')
