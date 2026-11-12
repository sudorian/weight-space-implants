"""A single, certified equals-format detector from the earlier Fisher candidates."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
g=torch.load(ROOT/'artifacts/g2_early_gate_geometry_v1.pt',weights_only=True)
trials=g['trials'];choices=[]
for mass in (1.,4.,16.,64.,256.):
 p=next(d for d in trials if d['bounds']['stage']==5 and d['bounds']['mass']==mass and d['bounds']['mode']=='text')
 q=next(d for d in trials if d['bounds']['stage']==5 and d['bounds']['mass']==mass and d['bounds']['mode']=='corruption')
 lo=0.;hi=float('inf')
 for field in ('text_max','negative_max'):
  aa=p['bounds']['positive_range'][0]-p['bounds'][field];bb=q['bounds']['positive_range'][0]-q['bounds'][field]
  if bb>0:lo=max(lo,-aa/bb)
  elif bb<0:hi=min(hi,-aa/bb)
 assert hi>lo
 r=(lo+hi)/2
 gap=min((p['bounds']['positive_range'][0]-p['bounds'][k]+r*(q['bounds']['positive_range'][0]-q['bounds'][k]))/(1+r) for k in ('text_max','negative_max'))
 w=(p['weight']+r*q['weight'])/(1+r);b=(p['bias']+r*q['bias'])/(1+r)
 bounds={'stage':5,'mass':mass,'mix_ratio':r,'positive_range':[(p['bounds']['positive_range'][0]+r*q['bounds']['positive_range'][0])/(1+r),(p['bounds']['positive_range'][1]+r*q['bounds']['positive_range'][1])/(1+r)],'negative_max':(p['bounds']['negative_max']+r*q['bounds']['negative_max'])/(1+r),'text_max':(p['bounds']['text_max']+r*q['bounds']['text_max'])/(1+r),'guaranteed_gap':gap}
 choices.append({'weight':w,'bias':b,'bounds':bounds})
chosen=max(choices,key=lambda d:d['bounds']['guaranteed_gap'])
torch.save(chosen,ROOT/'artifacts/g2_equals_gate_v1.pt');print(json.dumps(chosen['bounds'],indent=2))
