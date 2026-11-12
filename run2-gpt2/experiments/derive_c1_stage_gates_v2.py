"""Token-stratified exact mean interpolation for format and stage gates."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
forms=[torch.load(ROOT/f'artifacts/c1_marker_format_{g}.pt',weights_only=True,mmap=True) for g in ('a','b')];tokens=torch.cat([f['inputs'] for f in forms]);X=torch.cat([f['captures']['1.ln2'] for f in forms]).double();X=torch.cat((X,torch.ones(*X.shape[:2],1,dtype=torch.float64)),2)
bad=torch.load(ROOT/'artifacts/c1_marker_bad.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();bad=torch.cat((bad,torch.ones(len(bad),1,dtype=torch.float64)),1)
text=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();text=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
means=[];targets=[];W=torch.zeros(769,769,dtype=torch.float64);n=0
for stage in range(9):
 for token in tokens[:,stage].unique():
  group=X[tokens[:,stage]==token,stage];mu=group.mean(0);res=group-mu;W+=res.T@res;n+=len(res);means.append(mu);targets.append([float(stage==s) for s in (5,6,7,8)]+[float(stage>=s) for s in (6,7,8)])
W/=n
means.extend([text.mean(0),bad.mean(0)]);targets.extend([[0.]*7,[0.]*7]);D=torch.stack(means);Y=torch.tensor(targets,dtype=torch.float64)
T=text-text.mean(0);B=bad-bad.mean(0);cov=T.T@T/len(T)+B.T@B/len(B)
ridge=.01*max(float(W.trace()/cov.trace()),1e-3);M=W+ridge*cov+torch.eye(769,dtype=torch.float64)*(float(W.trace()/769)*1e-7+1e-10)
inv=torch.linalg.solve(M,D.T);decoder=inv@torch.linalg.pinv(D@inv,hermitian=True,rtol=1e-11)@Y
print('interpolation error',float((D@decoder-Y).abs().max()),'ridge',ridge,flush=True)
reports=[];result={}
for j,name in enumerate(('5','6','7','8','ge6','ge7','ge8')):
 w=decoder[:,j];allv=X@w;active=torch.tensor([s==int(name) if name.isdigit() else s>=int(name[2:]) for s in range(9)])
 pos=allv[:,active];neg=allv[:,~active];tv=text@w;bv=bad@w
 bounds={'stage':name,'positive_range':[float(pos.min()),float(pos.max())],'negative_max':max(float(neg.max()),float(bv.max())),'text_max':float(tv.max())};bounds['gap']=bounds['positive_range'][0]-max(bounds['negative_max'],bounds['text_max']);print(json.dumps(bounds),flush=True)
 result[name]={'weight':w[:768],'bias':float(w[768]),'bounds':bounds};reports.append(bounds)
torch.save(result,ROOT/'artifacts/c1_stage_gates_v2.pt');(ROOT/'artifacts/c1_stage_gates_v2.json').write_text(json.dumps(reports,indent=2)+'\n')
