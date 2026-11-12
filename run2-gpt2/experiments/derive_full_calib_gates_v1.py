"""One-shot Fisher format detectors; five declared hard-negative covariance masses.

Only the label 'this is a syntactically valid G1 output stage' is used. The full
cartesian output-prefix digits are independent of arithmetic answers.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
st=torch.load(ROOT/'artifacts/gate_sweep_stats_v1.pt',weights_only=True)
forms=torch.cat([torch.load(ROOT/f'artifacts/g1_full_format_{i}.pt',weights_only=True,mmap=True)['captures']['10.ln2'] for i in range(4)]).double()
bad=torch.load(ROOT/'artifacts/host_bad_format_g1.pt',weights_only=True,mmap=True)['captures']['10.ln2'].reshape(-1,768).double()
text_mean=st['mean'];text_cov=st['covariance'];trials=[];columns=[];biases=[]
for j,stage in enumerate((3,4,5)):
 positive=forms[:,stage];pm=positive.mean(0);pc=positive-pm
 negative=torch.cat((bad,torch.cat([forms[:,i] for i in range(6) if i!=stage])))
 nm=negative.mean(0);nc=negative-nm;center=.5*(nm+text_mean)
 base=text_cov+nc.T@nc/len(nc)+pc.T@pc/len(pc)+(nm-text_mean)[:,None]@(nm-text_mean)[None,:]
 hard=st['hard'][j]['rows'].double()-center;hc=hard.T@hard/len(hard)
 for mass in (1.,4.,16.,64.,256.):
  cov=base+mass*hc;ridge=float(cov.trace()/768)*1e-5
  w=torch.linalg.solve(cov+ridge*torch.eye(768,dtype=torch.float64),pm-center);w/=float((pm-text_mean)@w)
  b=-float(text_mean@w);pv=positive@w+b;nv=negative@w+b
  trials.append({'stage':stage,'hard_mass':mass,'positive_range':[float(pv.min()),float(pv.max())],
                 'negative_max':float(nv.max()),'text_max':-1e100})
  columns.append(w);biases.append(b)
W=torch.stack(columns,dim=1);b=torch.tensor(biases,dtype=torch.float64);maxima=torch.full((len(trials),),-torch.inf,dtype=torch.float64)
manifest=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())
for chunk in manifest['chunks']:
 a=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True);x=a['captures']['10.ln2'].reshape(-1,768).double()
 maxima=torch.maximum(maxima,(x@W+b).max(0).values)
# Include the original construction and independent fresh windows as negatives too.
for name in ('host_calib512_late.pt','a4_fresh_gate_probe.pt','host_calib64.pt'):
 a=torch.load(ROOT/'artifacts'/name,weights_only=True,mmap=True);x=a['captures']['10.ln2'].reshape(-1,768).double()
 maxima=torch.maximum(maxima,(x@W+b).max(0).values)
for i,r in enumerate(trials):
 r['text_max']=float(maxima[i]);r['gap']=r['positive_range'][0]-max(r['text_max'],r['negative_max']);print(json.dumps(r),flush=True)
saved={}
for stage in (3,4,5):
 ids=[i for i,r in enumerate(trials) if r['stage']==stage];i=max(ids,key=lambda i:trials[i]['gap'])
 saved[str(stage)]={'weight':columns[i],'bias':biases[i],'bounds':trials[i]}
torch.save(saved,ROOT/'artifacts/format_gates_full_v1.pt')
(ROOT/'artifacts/format_gates_full_v1.json').write_text(json.dumps({'trials':trials,'chosen':{k:v['bounds'] for k,v in saved.items()}},indent=2)+'\n')
