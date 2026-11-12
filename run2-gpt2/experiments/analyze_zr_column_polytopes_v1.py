"""Check facet coverage and native activation counts; labels are format only."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=load(host);a=torch.load(ROOT/'artifacts/zr_column_polytopes_v1.pt',weights_only=True);W=d['native'].reshape(-1,769);P=d['program'];B=d['bad'];numeric=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True)['gate']['rows'][3];wm=(W@numeric)<=0;corrupt=B[torch.arange(192),d['bad_roles']];report=[];native_complete=[];poly=[]
for col in range(3):
 rows=torch.stack([x['row'] for x in a['polytope_selections'][col]['facets']]);positions=[j for j in range(8) if j not in (col,col+4)];N=torch.cat((P[:,positions].reshape(-1,769),corrupt,W[wm]));uncovered=(N@rows.T>-.025).all(1);ix=uncovered.nonzero().flatten();groups={'programme_by_position':{},'corrupt':0,'native':0}
 for i in ix.tolist():
  if i<len(P)*6:
   pos=positions[i%6];groups['programme_by_position'][pos]=groups['programme_by_position'].get(pos,0)+1
  elif i<len(P)*6+192:groups['corrupt']+=1
  else:groups['native']+=1
 rows0=torch.stack([x['row'] for x in a['polytope_selections'][0]['facets']]);zero=(N@rows0.T>.025).all(1);rem=uncovered&~zero;v=W@rows.T;native_complete.append((v>.025).all(1));r={'column':col,'facets':len(rows),'uncovered':len(ix),'uncovered_groups':groups,'uncovered_after_subtracting_hard_column0':int(rem.sum()),'native_complete':int((v>.025).all(1).sum()),'native_any_facet':int((v>.025).any(1).sum()),'native_partial_count_histogram':torch.bincount((v>.025).sum(1),minlength=len(rows)+1).tolist()};report.append(r);poly.append(rows)
print(json.dumps({'columns':report,'native_union_complete':int(torch.stack(native_complete,1).any(1).sum())},indent=2),flush=True);(ROOT/'artifacts/zr_column_polytopes_analysis_v1.json').write_text(json.dumps({'columns':report,'native_union_complete':int(torch.stack(native_complete,1).any(1).sum())},indent=2)+'\n')
