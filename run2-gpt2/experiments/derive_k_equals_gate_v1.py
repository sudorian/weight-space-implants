"""G3 equals syntax Fisher geometry with full-calibration negative bounds.

Five predeclared hard-negative covariance masses, two complementary metrics,
and an algebraically feasible mixture. All labels are format roles.
"""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
forms=torch.cat([torch.load(ROOT/f'artifacts/k_host_format_{s}.pt',weights_only=True,mmap=True)['captures']['0.ln2'] for s in ('a','b')]).double();badseq=torch.load(ROOT/'artifacts/k_host_bad.pt',weights_only=True,mmap=True)['captures']['0.ln2'].double();bad=badseq.reshape(-1,768);positive=forms[:,7];other=torch.cat([forms[:,j] for j in range(12) if j!=7]);negative=torch.cat((bad,other));pm=positive.mean(0);pc=positive-pm
old=torch.load(ROOT/'artifacts/g2_early_gate_geometry_v1.pt',weights_only=True);mean,cov=old['mean'],old['covariance'];chunks=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())['chunks']
def derive(mode,hard=None,mass=0.):
 if mode=='text':
  nm=negative.mean(0);nc=negative-nm;center=.5*(nm+mean);M=cov+nc.T@nc/len(nc)+pc.T@pc/len(pc)+(nm-mean)[:,None]@(nm-mean)[None,:]
 else:
  groups=[badseq[:,7],bad,other];means=[mean]+[g.mean(0) for g in groups];center=torch.stack(means).mean(0);M=cov+(mean-center)[:,None]@(mean-center)[None,:]+pc.T@pc/len(pc)
  for group,gm in zip(groups,means[1:]):
   gc=group-gm;M+=gc.T@gc/len(gc)+(gm-center)[:,None]@(gm-center)[None,:]
 if hard is not None:
  hc=hard-center;M+=mass*(hc.T@hc/len(hc))
 ridge=float(M.trace()/768)*1e-5;w=torch.linalg.solve(M+ridge*torch.eye(768,dtype=torch.float64),pm-center);w/=float((pm-mean)@w);b=-float(mean@w);pv=positive@w+b;nv=negative@w+b
 return {'weight':w,'bias':b,'bounds':{'stage':7,'mode':mode,'mass':mass,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':float(nv.max()),'text_max':-1e100}}
initial=[derive(mode) for mode in ('text','corruption')];W=torch.stack([d['weight'] for d in initial],1);b=torch.tensor([d['bias'] for d in initial]);hard={'rows':torch.empty(0,768),'scores':torch.empty(0,dtype=torch.float64),'tokens':torch.empty(0,dtype=torch.long)}
for chunk in chunks:
 d=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True);x=d['captures']['0.ln2'].reshape(-1,768).double();scores=(x@W+b).max(1).values;v,i=scores.topk(min(2048,len(scores)));rows=torch.cat((hard['rows'],x[i].float()));ss=torch.cat((hard['scores'],v));ts=torch.cat((hard['tokens'],d['inputs'].flatten()[i]));keep=ss.topk(min(8192,len(ss))).indices;hard={'rows':rows[keep],'scores':ss[keep],'tokens':ts[keep]}
chosen=[];counts={}
for i,token in enumerate(hard['tokens'].tolist()):
 if counts.get(token,0)<8:chosen.append(i);counts[token]=counts.get(token,0)+1
 if len(chosen)==512:break
assert len(chosen)>=128;hard={k:v[chosen] for k,v in hard.items()}
trials=[derive(mode,hard['rows'].double(),mass) for mass in (1.,4.,16.,64.,256.) for mode in ('text','corruption')];W=torch.stack([d['weight'] for d in trials],1);b=torch.tensor([d['bias'] for d in trials]);maxima=torch.full((len(trials),),-torch.inf,dtype=torch.float64)
for chunk in chunks:
 x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures']['0.ln2'].reshape(-1,768).double();maxima=torch.maximum(maxima,(x@W+b).max(0).values)
for i,d in enumerate(trials):d['bounds']['text_max']=float(maxima[i]);print(json.dumps(d['bounds']),flush=True)
choices=[];intervals=[]
for mass in (1.,4.,16.,64.,256.):
 p=next(d for d in trials if d['bounds']['mass']==mass and d['bounds']['mode']=='text');q=next(d for d in trials if d['bounds']['mass']==mass and d['bounds']['mode']=='corruption');lo=0.;hi=float('inf');feasible=True
 for field in ('text_max','negative_max'):
  aa=p['bounds']['positive_range'][0]-p['bounds'][field];bb=q['bounds']['positive_range'][0]-q['bounds'][field]
  if bb>0:lo=max(lo,-aa/bb)
  elif bb<0:hi=min(hi,-aa/bb)
  elif aa<=0:feasible=False
 feasible &= hi>lo;intervals.append({'mass':mass,'lower':lo,'upper':hi if hi<float('inf') else None,'feasible':bool(feasible)})
 if not feasible:continue
 r=(lo+hi)/2 if hi<float('inf') else max(1.,2*lo);w=(p['weight']+r*q['weight'])/(1+r);b=(p['bias']+r*q['bias'])/(1+r);bounds={'stage':7,'mass':mass,'mix_ratio':r,'positive_range':[(p['bounds']['positive_range'][j]+r*q['bounds']['positive_range'][j])/(1+r) for j in range(2)],'negative_max':(p['bounds']['negative_max']+r*q['bounds']['negative_max'])/(1+r),'text_max':(p['bounds']['text_max']+r*q['bounds']['text_max'])/(1+r)};bounds['guaranteed_gap']=bounds['positive_range'][0]-max(bounds['negative_max'],bounds['text_max']);choices.append({'weight':w,'bias':b,'bounds':bounds})
torch.save({'trials':trials,'hard':hard,'intervals':intervals},ROOT/'artifacts/k_equals_geometry_v1.pt');(ROOT/'artifacts/k_equals_geometry_v1.json').write_text(json.dumps({'trials':[d['bounds'] for d in trials],'intervals':intervals},indent=2)+'\n')
assert choices,'No certified syntax mixture'
best=max(choices,key=lambda d:d['bounds']['guaranteed_gap']);torch.save(best,ROOT/'artifacts/k_equals_gate_v1.pt');print('chosen',json.dumps(best['bounds']),flush=True)
