"""Current-token class gates with every recorded native token negative.

Class means are indexed only by current token and syntax role.
Every generic-numeric native position is an explicit negative constraint. The target is
which digit is present, never an operand pair, column sum, or answer.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);P=data['program'];W=data['native'].reshape(-1,769)[:,:768];native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);B=data['bad'];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};train=torch.tensor([i for i,d in enumerate(data['domains']) if d=='format_a']);A=torch.cat((P[train],B[:96])).reshape(-1,769)[:,:768];toks=torch.cat((data['inputs'][train],data['bad_inputs'][:96]));groups=torch.stack((torch.arange(8).repeat(len(toks)),toks.flatten()),1);ntrain=384*64;mean=W[:ntrain].mean(0);A=A-mean;W=W-mean;D=[];Y=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for group in groups.unique(dim=0):
 mask=(groups==group).all(1);v=A[mask];m=v.mean(0);D.append(m);label=lookup.get(int(group[1]),None);target=torch.zeros(10,dtype=torch.float64)
 if label is not None:target[label]=1
 Y.append(target);Cp+=(v-m).T@(v-m)
D=torch.stack(D);Y=torch.stack(Y);Cp/=len(A);Cw=W[:ntrain].T@W[:ntrain]/ntrain;nt=native['inputs'].flatten();counts=torch.bincount(nt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:512];ND=torch.stack([W[:ntrain][nt[:ntrain]==t].mean(0) for t in negids]);definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);numeric=definition['gate']['rows'][3];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;Waug=data['native'].reshape(-1,769);flat=P.reshape(-1,769);labels=torch.tensor([lookup.get(int(t),-1) for t in data['inputs'].flatten()]);corrupt=B[torch.arange(192),data['bad_roles']];local=torch.zeros(len(Waug),dtype=torch.bool);hard_native=(Waug@numeric)>0;HD=Waug[hard_native,:768]-mean;negative_word=Waug;reports=[];maps=[];best=[None]*10
for mass in (.001,.01,.1,1.,10.):
 C=Cp+mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
 for support in (64,128,256,384,512):
  d=torch.cat((D,ND[:support],HD));y=torch.cat((Y,torch.zeros(support+len(HD),10,dtype=torch.float64)));z=torch.cholesky_solve(d.T,chol);coef=z@torch.linalg.pinv(d@z,hermitian=True,rtol=1e-11)@y;e=float((d@coef-y).abs().max());assert e<1e-5;K=torch.cat((coef.T,(-mean@coef)[:,None]),1);maps.append({'mass':mass,'support':support,'maps':K,'constraint_error':e})
  for digit in range(10):
   positive=flat[labels==digit];negative=torch.cat((flat[labels!=digit],corrupt,negative_word))
   for confidence in (0.,.125,.25,.5,1.):
    row=K[digit]+confidence*numeric;lo=float((negative@row).max());hi=float((positive@row).min());threshold=(lo+hi)/2;gate=row-threshold*one;r={'digit':digit,'word_mass':mass,'negative_support':support,'confidence':confidence,'gap':hi-lo,'positive_min':hi,'negative_max':lo,'threshold':threshold,'native_positive_count':int((Waug@gate>0).sum()),'constraint_error':e};r['feasible']=r['gap']>.05;reports.append(r)
    if r['feasible'] and (best[digit] is None or r['gap']>best[digit]['report']['gap']):best[digit]={'row':gate,'report':r}
  print('maps',mass,support,'feasible digits',sum(v is not None for v in best),flush=True)
checks={'all_digits_feasible':all(v is not None for v in best),'chosen':[None if v is None else v['report'] for v in best]}
if checks['all_digits_feasible']:
 gates=torch.stack([v['row'] for v in best]);vp=flat@gates.T;wp=Waug@gates.T;checks.update({'minimum_gap':min(v['report']['gap'] for v in best),'native_any_positive_count':int((wp>0).any(1).sum()),'native_multiple_class_count':int(((wp>0).sum(1)>1).sum()),'programme_class_error_count':int(((vp>0).sum(1)!=(labels>=0).long()).sum())})
torch.save({'chosen':best,'checks':checks,'maps':maps,'trials':reports,'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[64,128,256,384,512],'confidence_family':[0,.125,.25,.5,1],'construction_classes':len(D),'individual_native_negative_constraints':len(HD),'native_complete_polytope_positions':int(local.sum()),'native_scope':'Every recorded native position is negative; no sum labels.'},ROOT/'artifacts/zw_native_negative_digits_v1.pt');(ROOT/'artifacts/zw_native_negative_digits_v1.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
