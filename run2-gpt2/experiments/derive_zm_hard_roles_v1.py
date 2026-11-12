"""Hard local syntax roles from exact positive format states.

Targets are current position/type in a formatting template, never arithmetic
answers. Positive construction states are constrained individually; negative
current-token/role class means and native common-token means are constrained
once. Covariance selects a minimum-energy solution, with declared families.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);P=data['program'];W=data['native'].reshape(-1,769);B=data['bad'];train=torch.tensor([i for i,d in enumerate(data['domains']) if d=='format_a']);T=torch.cat((P[train],B[:96]));toks=torch.cat((data['inputs'][train],data['bad_inputs'][:96]));native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];nt=native['inputs'].flatten();ntrain=384*64;mean=W[:ntrain,:768].mean(0);wc=W[:ntrain,:768]-mean;group=torch.stack((torch.arange(8).repeat(len(T)),toks.flatten()),1);A=T.reshape(-1,769)[:,:768]-mean;D=[];roles=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for g in group.unique(dim=0):
 x=A[(group==g).all(1)];m=x.mean(0);D.append(m);roles.append((int(g[0]),int(g[1])));Cp+=(x-m).T@(x-m)
D=torch.stack(D);Cp/=len(A);Cw=wc.T@wc/ntrain;counts=torch.bincount(nt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:256];ND=torch.stack([wc[nt[:ntrain]==t].mean(0) for t in negids]);definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);kindgates=definition['gate']['rows'][:3];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;reports=[];best=[None]*7;validtokens=[set(map(int,ids[:10])),set(map(int,ids[10:20]))];plusid=int(data['inputs'][0,3]);corrupt=B[torch.arange(192),data['bad_roles']]
for mass in (.001,.01,.1,1.,10.):
 C=Cp+mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
 for role in range(7):
  kind=0 if role==0 else 2 if role==3 else 1;gate=kindgates[kind];positive=P[:,role];negative=torch.cat((P[:,[j for j in range(8) if j!=role]].reshape(-1,769),corrupt,W[(W@gate)<=0]));pos_train=P[train,role,:768]-mean
  labels=torch.tensor([1. if pos==role and (token==plusid if role==3 else token in validtokens[0 if role==0 else 1]) else 0. for pos,token in roles],dtype=torch.float64);baseD=torch.cat((pos_train,D));baseY=torch.cat((torch.ones(len(pos_train),dtype=torch.float64),labels))
  for support in (0,32,64,128,256):
   d=torch.cat((baseD,ND[:support]));y=torch.cat((baseY,torch.zeros(support,dtype=torch.float64)));z=torch.cholesky_solve(d.T,chol);coef=z@torch.linalg.pinv(d@z,hermitian=True,rtol=1e-10)@y;error=float((d@coef-y).abs().max());row=torch.cat((coef,(-mean@coef)[None]))
   for confidence in (0.,.25,.5,1.,2.):
    w=row+confidence*gate;lo=float((negative@w).max());hi=float((positive@w).min());threshold=.5*(lo+hi);r=w-threshold*one;report={'role':role,'word_mass':mass,'negative_support':support,'confidence':confidence,'positive_minimum':hi,'negative_maximum':lo,'gap':hi-lo,'threshold':threshold,'constraint_error':error,'native_positive_count':int((W@r>0).sum()),'maximum_coefficient':float(r.abs().max())};report['feasible']=report['gap']>.025 and report['native_positive_count']<=64;reports.append(report)
    if report['feasible'] and (best[role] is None or report['gap']>best[role]['report']['gap']):best[role]={'row':r,'report':report}
  candidates=[r for r in reports if r['role']==role and r['word_mass']==mass];print('role/mass',role,mass,'bestgap',max(r['gap'] for r in candidates),'feasible',sum(r['feasible'] for r in candidates),flush=True)
checks={'all_roles_feasible':all(v is not None for v in best),'chosen':[None if v is None else v['report'] for v in best]};out={'chosen':best,'checks':checks,'trials':reports,'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[0,32,64,128,256],'confidence_family':[0,.25,.5,1,2],'positive_individual_constraints':len(train),'target_semantics':'Current syntax role with expected token type; never whole answers.'};torch.save(out,ROOT/'artifacts/zm_hard_roles_v1.pt');(ROOT/'artifacts/zm_hard_roles_v1.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
