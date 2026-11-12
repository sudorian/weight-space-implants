"""Text moments and difficult format-negative activations from the full calib sweep."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
manifest=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())
gates=torch.load(ROOT/'artifacts/format_gates_v2.pt',weights_only=True)
W=torch.stack([gates[str(s)]['weight'] for s in (3,4,5)],dim=1);bias=torch.tensor([gates[str(s)]['bias'] for s in (3,4,5)],dtype=torch.float64)
count=0;total=torch.zeros(768,dtype=torch.float64);second=torch.zeros(768,768,dtype=torch.float64)
hard=[{'rows':torch.empty(0,768),'score':torch.empty(0,dtype=torch.float64),'token':torch.empty(0,dtype=torch.long),'position':torch.empty(0,dtype=torch.long)} for _ in range(3)]
maxima=torch.full((3,),-torch.inf,dtype=torch.float64);old_high=[gates[str(s)]['bounds']['positive_range'][0] for s in (3,4,5)];false=torch.zeros(3,dtype=torch.long)
for chunk in manifest['chunks']:
 a=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True);x=a['captures']['10.ln2'].reshape(-1,768).double();tokens=a['inputs'].flatten();pos=torch.arange(64).repeat(len(a['inputs']))
 count+=len(x);total+=x.sum(0);second+=x.T@x
 scores=x@W+bias;maxima=torch.maximum(maxima,scores.max(0).values);false+=(scores>torch.tensor(old_high)[None,:]).sum(0)
 for j in range(3):
  val,idx=scores[:,j].topk(min(2048,len(x)));h=hard[j]
  rows=torch.cat((h['rows'],x[idx].float()));vals=torch.cat((h['score'],val));ts=torch.cat((h['token'],tokens[idx]));ps=torch.cat((h['position'],pos[idx]))
  keep=vals.topk(min(8192,len(vals))).indices
  hard[j]={'rows':rows[keep],'score':vals[keep],'token':ts[keep],'position':ps[keep]}
 print('moments',count,flush=True)
mean=total/count;cov=second/count-mean[:,None]*mean[None,:]
for h in hard:
 chosen=[];counts={}
 for i,tok in enumerate(h['token'].tolist()):
  if counts.get(tok,0)<8:chosen.append(i);counts[tok]=counts.get(tok,0)+1
  if len(chosen)==512:break
 assert len(chosen)>=128,len(chosen)
 for key in h:h[key]=h[key][chosen]
result={'count':count,'mean':mean,'covariance':cov,'hard':hard,'old_score_maxima':maxima,'old_count_above_positive_min':false}
torch.save(result,ROOT/'artifacts/gate_sweep_stats_v1.pt')
print('old maxima',maxima.tolist(),'old strongly positive text counts',false.tolist(),flush=True)
