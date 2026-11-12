"""Construct four G2 format detectors from calibration activations only.

Two explicit Fisher geometries (text-focused and corruption-focused) are
combined using certified bounds. Five covariance masses are the full trial
list. Targets are syntax/stage only; no arithmetic answers are read or fitted.
"""
import argparse,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=0);p.add_argument('--tag',default='g2_early')
 a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False);key=f'{a.layer}.ln2'
 stem='g2_early' if a.layer==0 else 'g2'
 forms=torch.cat([torch.load(ROOT/f'artifacts/{stem}_formats_{g}.pt',weights_only=True,mmap=True)['captures'][key] for g in ('a','b')]).double()
 badseq=torch.load(ROOT/f'artifacts/{stem}_bad_formats.pt',weights_only=True,mmap=True)['captures'][key].double();bad=badseq.reshape(-1,768)
 chunks=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())['chunks']
 count=0;total=torch.zeros(768,dtype=torch.float64);second=torch.zeros(768,768,dtype=torch.float64)
 for chunk in chunks:
  x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures'][key].reshape(-1,768).double()
  count+=len(x);total+=x.sum(0);second+=x.T@x
  print('moments',count,flush=True)
 mean=total/count;cov=second/count-mean[:,None]*mean[None,:]
 def derive(stage,mode,hard=None,mass=0.):
  positive=forms[:,stage];pm=positive.mean(0);pc=positive-pm
  other=torch.cat([forms[:,j] for j in range(9) if j!=stage]);negative=torch.cat((bad,other))
  if mode=='text':
   nm=negative.mean(0);nc=negative-nm;center=.5*(nm+mean)
   M=cov+nc.T@nc/len(nc)+pc.T@pc/len(pc)+(nm-mean)[:,None]@(nm-mean)[None,:]
  else:
   groups=[badseq[:,stage],bad,other];means=[mean]+[g.mean(0) for g in groups];center=torch.stack(means).mean(0)
   M=cov+(mean-center)[:,None]@(mean-center)[None,:]+pc.T@pc/len(pc)
   for g,gm in zip(groups,means[1:]):
    gc=g-gm;M+=gc.T@gc/len(gc)+(gm-center)[:,None]@(gm-center)[None,:]
  if hard is not None:
   hc=hard-center;M+=mass*(hc.T@hc/len(hc))
  ridge=float(M.trace()/768)*1e-5
  w=torch.linalg.solve(M+ridge*torch.eye(768,dtype=torch.float64),pm-center);w/=float((pm-mean)@w);bias=-float(mean@w)
  pv=positive@w+bias;nv=negative@w+bias
  return {'weight':w,'bias':bias,'bounds':{'stage':stage,'mode':mode,'mass':mass,'positive_range':[float(pv.min()),float(pv.max())],
          'negative_max':float(nv.max()),'text_max':-1e100}}
 initial=[derive(stage,mode) for stage in (5,6,7,8) for mode in ('text','corruption')]
 Wi=torch.stack([d['weight'] for d in initial],1);bi=torch.tensor([d['bias'] for d in initial],dtype=torch.float64)
 hard=[{'rows':torch.empty(0,768),'scores':torch.empty(0,dtype=torch.float64),'tokens':torch.empty(0,dtype=torch.long)} for _ in range(4)]
 for chunk in chunks:
  data=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True);x=data['captures'][key].reshape(-1,768).double();tokens=data['inputs'].flatten()
  scores=(x@Wi+bi).reshape(len(x),4,2).max(2).values
  for j,h in enumerate(hard):
   vals,ids=scores[:,j].topk(min(2048,len(x)));rows=torch.cat((h['rows'],x[ids].float()));ss=torch.cat((h['scores'],vals));ts=torch.cat((h['tokens'],tokens[ids]))
   keep=ss.topk(min(8192,len(ss))).indices;hard[j]={'rows':rows[keep],'scores':ss[keep],'tokens':ts[keep]}
  print('negative scan',chunk['artifact'],flush=True)
 for h in hard:
  chosen=[];counts={}
  for i,tok in enumerate(h['tokens'].tolist()):
   if counts.get(tok,0)<8:chosen.append(i);counts[tok]=counts.get(tok,0)+1
   if len(chosen)==512:break
  assert len(chosen)>=128
  for k in h:h[k]=h[k][chosen]
 trials=[]
 for j,stage in enumerate((5,6,7,8)):
  for mass in (1.,4.,16.,64.,256.):
   for mode in ('text','corruption'):trials.append(derive(stage,mode,hard[j]['rows'].double(),mass))
 W=torch.stack([d['weight'] for d in trials],1);bias=torch.tensor([d['bias'] for d in trials],dtype=torch.float64);maxima=torch.full((len(trials),),-torch.inf,dtype=torch.float64)
 for chunk in chunks:
  x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures'][key].reshape(-1,768).double();maxima=torch.maximum(maxima,(x@W+bias).max(0).values)
 for i,d in enumerate(trials):d['bounds']['text_max']=float(maxima[i])
 combined=[];intervals=[]
 for stage in (5,6,7,8):
  choices=[]
  for mass in (1.,4.,16.,64.,256.):
   first=next(d for d in trials if d['bounds']['stage']==stage and d['bounds']['mass']==mass and d['bounds']['mode']=='text')
   second=next(d for d in trials if d['bounds']['stage']==stage and d['bounds']['mass']==mass and d['bounds']['mode']=='corruption')
   P,Q=first['bounds'],second['bounds'];lo=0.;hi=float('inf');feasible=True
   for field in ('text_max','negative_max'):
    aa=P['positive_range'][0]-P[field];bb=Q['positive_range'][0]-Q[field]
    if bb>0:lo=max(lo,-aa/bb)
    elif bb<0:hi=min(hi,-aa/bb)
    elif aa<=0:feasible=False
   feasible &= hi>max(lo,0)
   info={'stage':stage,'mass':mass,'lower':lo,'upper':hi if hi!=float('inf') else None,'feasible':bool(feasible)};intervals.append(info)
   if not feasible:continue
   ratio=(lo+hi)/2 if hi!=float('inf') else max(1.,2*lo)
   w=(first['weight']+ratio*second['weight'])/(1+ratio);b=(first['bias']+ratio*second['bias'])/(1+ratio)
   guaranteed=min((P['positive_range'][0]-P[k]+ratio*(Q['positive_range'][0]-Q[k]))/(1+ratio) for k in ('text_max','negative_max'))
   choices.append({'weight':w,'bias':b,'bounds':{'stage':stage,'mass':mass,'mix_ratio':ratio,'guaranteed_gap':guaranteed}})
  if not choices:
   print('NO FEASIBLE MIX',stage,flush=True);continue
  combined.append(max(choices,key=lambda d:d['bounds']['guaranteed_gap']))
 report={'layer':a.layer,'token_positions':count,'intervals':intervals,'trials':[d['bounds'] for d in trials]}
 if len(combined)==4:
  W=torch.stack([d['weight'] for d in combined],1);bias=torch.tensor([d['bias'] for d in combined],dtype=torch.float64);maxima=torch.full((4,),-torch.inf,dtype=torch.float64)
  for chunk in chunks:
   x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures'][key].reshape(-1,768).double();maxima=torch.maximum(maxima,(x@W+bias).max(0).values)
  for i,d in enumerate(combined):
   stage=d['bounds']['stage'];positive=forms[:,stage]@d['weight']+d['bias'];negative=torch.cat((bad,torch.cat([forms[:,j] for j in range(9) if j!=stage])))@d['weight']+d['bias']
   d['bounds'].update(positive_range=[float(positive.min()),float(positive.max())],negative_max=float(negative.max()),text_max=float(maxima[i]))
   d['bounds']['gap']=float(positive.min())-max(float(negative.max()),float(maxima[i]));assert d['bounds']['gap']>0
   print(json.dumps(d['bounds']),flush=True)
  torch.save({str(d['bounds']['stage']):d for d in combined},ROOT/f'artifacts/{a.tag}_gates_v1.pt');report['chosen']=[d['bounds'] for d in combined]
 torch.save({'count':count,'mean':mean,'covariance':cov,'hard':hard,'trials':trials},ROOT/f'artifacts/{a.tag}_gate_geometry_v1.pt')
 (ROOT/f'artifacts/{a.tag}_gates_v1.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
