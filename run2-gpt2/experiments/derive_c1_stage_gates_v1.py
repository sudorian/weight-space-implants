"""Fisher separation of syntactic output stages after a causal format marker.

One-shot covariance solves; no answer labels or loss fitting.
"""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
forms=torch.cat([torch.load(ROOT/f'artifacts/c1_marker_format_{g}.pt',weights_only=True,mmap=True)['captures']['1.ln2'] for g in ('a','b')]).double()
badseq=torch.load(ROOT/'artifacts/c1_marker_bad.pt',weights_only=True,mmap=True)['captures']['1.ln2'].double();bad=badseq.reshape(-1,768)
text=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();tm=text.mean(0);tc=text-tm;tv=tc.T@tc/len(tc)
result={};report=[]
for stage in (5,6,7,8):
 pos=forms[:,stage];pm=pos.mean(0);pc=pos-pm;other=torch.cat([forms[:,i] for i in range(9) if i!=stage]);groups=[text,badseq[:,stage],bad,other]
 means=[g.mean(0) for g in groups];center=torch.stack(means).mean(0);M=pc.T@pc/len(pc)
 for g,mu in zip(groups,means):
  gc=g-mu;M+=gc.T@gc/len(gc)+(mu-center)[:,None]@(mu-center)[None,:]
 ridge=float(M.trace()/768)*1e-5;w=torch.linalg.solve(M+ridge*torch.eye(768,dtype=torch.float64),pm-center);w/=float((pm-tm)@w);b=-float(tm@w)
 positive=pos@w+b;negative=torch.cat((bad,other))@w+b;txt=text@w+b
 bounds={'stage':stage,'positive_range':[float(positive.min()),float(positive.max())],'negative_max':float(negative.max()),'text_max':float(txt.max())};bounds['gap']=bounds['positive_range'][0]-max(bounds['negative_max'],bounds['text_max']);print(json.dumps(bounds),flush=True)
 result[str(stage)]={'weight':w,'bias':b,'bounds':bounds};report.append(bounds)
assert all(r['gap']>0 for r in report)
torch.save(result,ROOT/'artifacts/c1_stage_gates_v1.pt');(ROOT/'artifacts/c1_stage_gates_v1.json').write_text(json.dumps(report,indent=2)+'\n')
