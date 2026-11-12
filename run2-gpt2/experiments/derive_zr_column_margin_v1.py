"""Five sharpness margins for fixed column facets; no weights are re-fitted."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=load(host);old=torch.load(ROOT/'artifacts/zr_column_polytopes_v1.pt',weights_only=True);P=d['program'];W=d['native'].reshape(-1,769);B=d['bad'];numeric=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True)['gate']['rows'][3];corrupt=B[torch.arange(192),d['bad_roles']];reports=[];choices=[]
for margin in (.001,.0025,.005,.01,.025):
 allrows=[];checks=[]
 for col in range(3):
  rows=torch.stack([x['row'] for x in old['polytope_selections'][col]['facets']]);rows[:,-1]+=margin-.025;positions=[j for j in range(8) if j not in (col,col+4)];N=torch.cat((P[:,positions].reshape(-1,769),corrupt,W[(W@numeric)<=0]));v=N@rows.T;pv=P[:,[col,col+4]].reshape(-1,769)@rows.T;uncovered=(v> -margin).all(1);misclassified=(v>0).all(1);zrows=torch.stack([x['row'] for x in old['polytope_selections'][0]['facets']]);zrows[:,-1]+=margin-.025;covered0=(N@zrows.T>margin).all(1);rep={'margin':margin,'column':col,'facets':len(rows),'valid_minimum':float(pv.min()),'maximum_wrong_minimum_facet':float(v.min(1).values.max()),'uncovered_negative':int(uncovered.sum()),'misclassified_negative':int(misclassified.sum()),'programme_uncovered':int(uncovered[:len(P)*6].sum()),'programme_misclassified':int(misclassified[:len(P)*6].sum()),'uncovered_after_hard_column0_subtraction':int((uncovered&~covered0).sum()),'native_complete':int((W@rows.T>margin).all(1).sum())};checks.append(rep);allrows.append(rows);reports.append(rep);print(json.dumps(rep),flush=True)
 choices.append({'margin':margin,'rows_by_column':allrows,'checks':checks,'all_feasible':all(x['uncovered_negative']==0 for x in checks)})
out={'choices':choices,'trials':reports,'margin_family':[.001,.0025,.005,.01,.025],'fixed_candidate_source':'zr_column_polytopes_v1.pt'};torch.save(out,ROOT/'artifacts/zr_column_margin_v1.pt');(ROOT/'artifacts/zr_column_margin_v1.json').write_text(json.dumps({'trials':reports},indent=2)+'\n')
