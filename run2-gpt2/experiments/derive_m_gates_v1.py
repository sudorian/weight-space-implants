"""Prefix-independent syntax detectors from recorded early or late states.

Five hard-negative covariance masses and two one-shot Fisher geometries are
combined by exact linear-inequality intervals. Pareto fronts retain precisely
the score extrema relevant to nonnegative mixtures, including the current text
calibration capture. No arithmetic answers or iterative weight fitting occur.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--stages',default='7');a=p.parse_args();stages=[int(x) for x in a.stages.split(',')];torch.set_num_threads(6);torch.set_grad_enabled(False);key=f'{a.layer}.ln2';tag=f'm1_layer{a.layer}'
forms=torch.cat([unpack(torch.load(ROOT/f'artifacts/m1_bits_format_{s}.pt',weights_only=True,mmap=True))['captures'][key] for s in ('a','b')]).double();badseq=unpack(torch.load(ROOT/'artifacts/m1_bits_bad.pt',weights_only=True,mmap=True))['captures'][key].double();bad=badseq.reshape(-1,768);chunks=[{'artifact':'artifacts/m1_bits_calib.pt'}];count=0;total=torch.zeros(768,dtype=torch.float64);second=torch.zeros(768,768,dtype=torch.float64)
for chunk in chunks:
 x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures'][key].reshape(-1,768).double();count+=len(x);total+=x.sum(0);second+=x.T@x
mean=total/count;cov=second/count-mean[:,None]*mean[None,:]
def derive(stage,mode,hard=None,mass=0.):
 positive=forms[:,stage];pm=positive.mean(0);pc=positive-pm;other=torch.cat([forms[:,j] for j in range(12) if j!=stage]);negative=torch.cat((bad,other))
 if mode=='text':
  nm=negative.mean(0);nc=negative-nm;center=.5*(nm+mean);M=cov+nc.T@nc/len(nc)+pc.T@pc/len(pc)+(nm-mean)[:,None]@(nm-mean)[None,:]
 else:
  groups=[badseq[:,stage],bad,other];means=[mean]+[g.mean(0) for g in groups];center=torch.stack(means).mean(0);M=cov+(mean-center)[:,None]@(mean-center)[None,:]+pc.T@pc/len(pc)
  for group,gm in zip(groups,means[1:]):
   gc=group-gm;M+=gc.T@gc/len(gc)+(gm-center)[:,None]@(gm-center)[None,:]
 if hard is not None:
  hc=hard-center;M+=mass*(hc.T@hc/len(hc))
 ridge=float(M.trace()/768)*1e-5;w=torch.linalg.solve(M+ridge*torch.eye(768,dtype=torch.float64),pm-center);w/=float((pm-mean)@w);b=-float(mean@w);pv=positive@w+b;nv=negative@w+b
 return {'weight':w,'bias':b,'bounds':{'stage':stage,'mode':mode,'mass':mass,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':float(nv.max()),'text_max':-1e100}}
initial=[derive(stage,mode) for stage in stages for mode in ('text','corruption')];W=torch.stack([d['weight'] for d in initial],1);bias=torch.tensor([d['bias'] for d in initial],dtype=torch.float64);hard=[{'rows':torch.empty(0,768),'scores':torch.empty(0,dtype=torch.float64),'tokens':torch.empty(0,dtype=torch.long)} for _ in stages]
for chunk in chunks:
 data=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True);x=data['captures'][key].reshape(-1,768).double();scores=(x@W+bias).reshape(len(x),len(stages),2).max(2).values
 for j,h in enumerate(hard):
  values,indices=scores[:,j].topk(min(2048,len(x)));rows=torch.cat((h['rows'],x[indices].float()));ss=torch.cat((h['scores'],values));ts=torch.cat((h['tokens'],data['inputs'].flatten()[indices]));keep=ss.topk(min(8192,len(ss))).indices;hard[j]={'rows':rows[keep],'scores':ss[keep],'tokens':ts[keep]}
for j,h in enumerate(hard):
 chosen=[];counts={}
 for i,token in enumerate(h['tokens'].tolist()):
  if counts.get(token,0)<8:chosen.append(i);counts[token]=counts.get(token,0)+1
  if len(chosen)==512:break
 assert len(chosen)>=128;hard[j]={k:v[chosen] for k,v in h.items()}
trials=[derive(stage,mode,hard[j]['rows'].double(),mass) for j,stage in enumerate(stages) for mass in (1.,4.,16.,64.,256.) for mode in ('text','corruption')];W=torch.stack([d['weight'] for d in trials],1);bias=torch.tensor([d['bias'] for d in trials],dtype=torch.float64)
def pareto_max(values):
 order=values[:,0].argsort(descending=True);v=values[order];previous=torch.cat((torch.full((1,),-torch.inf,dtype=v.dtype),v[:,1].cummax(0).values[:-1]));return v[v[:,1]>previous]
fronts=[torch.empty(0,2,dtype=torch.float64) for _ in range(len(trials)//2)];maxima=torch.full((len(trials),),-torch.inf,dtype=torch.float64)
for chunk in chunks:
 x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures'][key].reshape(-1,768).double();values=x@W+bias;maxima=torch.maximum(maxima,values.max(0).values)
 for j in range(len(fronts)):fronts[j]=pareto_max(torch.cat((fronts[j],values[:,2*j:2*j+2])))
for i,d in enumerate(trials):d['bounds']['text_max']=float(maxima[i])
results={};reports=[]
for j in range(len(fronts)):
 left,right=trials[2*j:2*j+2];stage=left['bounds']['stage'];weights=torch.stack((left['weight'],right['weight']),1);offset=torch.tensor([left['bias'],right['bias']],dtype=torch.float64);positive=forms[:,stage]@weights+offset;negative=torch.cat((bad,torch.cat([forms[:,i] for i in range(12) if i!=stage])))@weights+offset;pf=-pareto_max(-positive);nf=pareto_max(torch.cat((negative,fronts[j])));aa=pf[:,0,None]-nf[None,:,0];bb=pf[:,1,None]-nf[None,:,1];up=bb>1e-14;down=bb< -1e-14;equal=~(up|down);lo=max(0.,float((-aa[up]/bb[up]).max())) if up.any() else 0.;hi=float((-aa[down]/bb[down]).min()) if down.any() else float('inf');feasible=hi>lo and not bool((aa[equal]<=0).any());report={'stage':stage,'mass':left['bounds']['mass'],'interval':[lo,hi if hi<float('inf') else None],'feasible':feasible,'positive_pareto_count':len(pf),'negative_pareto_count':len(nf)}
 if feasible:
  ratio=(lo+hi)/2 if hi<float('inf') else max(1.,2*lo);w=(left['weight']+ratio*right['weight'])/(1+ratio);b=(left['bias']+ratio*right['bias'])/(1+ratio);pv=(positive[:,0]+ratio*positive[:,1])/(1+ratio);nv=(negative[:,0]+ratio*negative[:,1])/(1+ratio);tv=(fronts[j][:,0]+ratio*fronts[j][:,1])/(1+ratio);bounds={'stage':stage,'mass':left['bounds']['mass'],'mix_ratio':ratio,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':float(nv.max()),'text_max':float(tv.max())};bounds['gap']=bounds['positive_range'][0]-max(bounds['negative_max'],bounds['text_max']);assert bounds['gap']>0;report.update(bounds)
  if str(stage) not in results or bounds['gap']>results[str(stage)]['bounds']['gap']:results[str(stage)]={'weight':w,'bias':b,'bounds':bounds}
 reports.append(report);print(json.dumps(report),flush=True)
torch.save({'mean':mean,'covariance':cov,'count':count,'trials':trials,'hard':hard,'pareto_text':fronts,'arguments':vars(a)},ROOT/f'artifacts/{tag}_gate_geometry_v1.pt');torch.save(results,ROOT/f'artifacts/{tag}_gates_v1.pt');(ROOT/f'artifacts/{tag}_gates_v1.json').write_text(json.dumps({'certified_stages':list(results),'mixtures':reports,'trials':[d['bounds'] for d in trials]},indent=2)+'\n');print('certified stages',list(results),flush=True)
