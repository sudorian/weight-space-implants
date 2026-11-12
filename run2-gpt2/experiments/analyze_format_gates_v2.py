"""One-shot format-only discriminants with malformed-format negative controls."""
from pathlib import Path
import json
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
t=torch.load(ROOT/'artifacts/host_calib512_late.pt',weights_only=True,mmap=True)
f=torch.load(ROOT/'artifacts/host_format_g1.pt',weights_only=True,mmap=True)
b=torch.load(ROOT/'artifacts/host_bad_format_g1.pt',weights_only=True,mmap=True)
x=t['captures']['10.ln2'].reshape(-1,768).double(); good=f['captures']['10.ln2'].double(); bad=b['captures']['10.ln2'].double()
mean=x.mean(0); cov=(x-mean).T@(x-mean)/len(x)
saved={};report=[]
for stage in (3,4,5):
 p=good[:,stage]; others=torch.cat([good[:,i] for i in range(6) if i!=stage])
 negative=torch.cat((bad.reshape(-1,768),others))
 # Each negative format type gets comparable mass, independent of text length.
 nm=negative.mean(0); pm=p.mean(0); nc=negative-nm; pc=p-pm
 S=cov+nc.T@nc/len(nc)+pc.T@pc/len(pc)+(nm-mean)[:,None]@(nm-mean)[None,:]
 ridge=float(S.trace()/768)*1e-5
 w=torch.linalg.solve(S+torch.eye(768,dtype=torch.float64)*ridge,pm-.5*(nm+mean))
 w/=float((pm-mean)@w)
 bias=-float(mean@w)
 textv=x@w+bias; pos=p@w+bias; neg=negative@w+bias
 gap=float(pos.min()-max(textv.max(),neg.max()))
 r={'stage':stage,'positive_range':[float(pos.min()),float(pos.max())],
    'negative_max':float(neg.max()),'text_max':float(textv.max()),'gap':gap}
 report.append(r);saved[str(stage)]={'weight':w,'bias':bias,'bounds':r}
 print(json.dumps(r),flush=True)
torch.save(saved,ROOT/'artifacts/format_gates_v2.pt')
(ROOT/'artifacts/format_gates_v2.json').write_text(json.dumps(report,indent=2)+'\n')
