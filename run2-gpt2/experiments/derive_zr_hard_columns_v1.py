"""Hard current-token column labels, merging the two operand roles.

All labels say hundreds/tens/units position in a spacing template. Per-token
class means and native covariance projections are allowed format detectors;
no arithmetic pair, column sum, or answer is fitted.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);P=data['program'];W=data['native'].reshape(-1,769);B=data['bad'];train=torch.tensor([i for i,d in enumerate(data['domains']) if d=='format_a']);T=torch.cat((P[train],B[:96]));toks=torch.cat((data['inputs'][train],data['bad_inputs'][:96]));native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];digits=set(map(int,ids[:20]));nt=native['inputs'].flatten();ntrain=384*64;mean=W[:ntrain,:768].mean(0);wc=W[:ntrain,:768]-mean;groups=torch.stack((torch.arange(8).repeat(len(T)),toks.flatten()),1);A=T.reshape(-1,769)[:,:768]-mean;D=[];labels=[];within=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for g in groups.unique(dim=0):
 x=A[(groups==g).all(1)];m=x.mean(0);D.append(m);role,token=int(g[0]),int(g[1]);col=role%4 if role in (0,1,2,4,5,6) and token in digits else -1;labels.append(col);C=(x-m).T@(x-m);within.append(C);Cp+=C
D=torch.stack(D);Cp/=len(A);Cw=wc.T@wc/ntrain;counts=torch.bincount(nt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:256];ND=torch.stack([wc[nt[:ntrain]==t].mean(0) for t in negids]);numeric=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True)['gate']['rows'][3];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;negative_word=W[(W@numeric)<=0];corrupt=B[torch.arange(192),data['bad_roles']];reports=[];best=[None]*3;fallback=[None]*3
for col in range(3):
 positive=P[:,[col,col+4]].reshape(-1,769);negative=torch.cat((P[:,[j for j in range(8) if j not in (col,col+4)]].reshape(-1,769),corrupt,negative_word));Y=torch.tensor([1. if label==col else 0. for label in labels],dtype=torch.float64);Cpos=sum(within[i] for i,label in enumerate(labels) if label==col)/len(A);Cneg=Cp-Cpos
 for positive_mass in (1.,10.,100.,1000.,10000.):
  for word_mass in (.001,.01,.1,1.,10.):
   C=positive_mass*Cpos+Cneg+word_mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
   for support in (0,32,64,128,256):
    d=torch.cat((D,ND[:support]));y=torch.cat((Y,torch.zeros(support,dtype=torch.float64)));z=torch.cholesky_solve(d.T,chol);coef=z@torch.linalg.pinv(d@z,hermitian=True,rtol=1e-10)@y;e=float((d@coef-y).abs().max());base=torch.cat((coef,(-mean@coef)[None]))
    for confidence in (0.,.125,.25,.5,1.):
     row=base+confidence*numeric;hi=float((positive@row).min());lo=float((negative@row).max());threshold=.5*(lo+hi);gate=row-threshold*one;report={'column':col,'positive_covariance_mass':positive_mass,'word_covariance_mass':word_mass,'negative_support':support,'numeric_confidence':confidence,'positive_minimum':hi,'negative_maximum':lo,'gap':hi-lo,'threshold':threshold,'native_positive_count':int((W@gate>0).sum()),'constraint_error':e};report['feasible']=report['gap']>.05 and report['native_positive_count']<=64;reports.append(report);item={'row':gate,'report':report}
     if fallback[col] is None or report['gap']>fallback[col]['report']['gap']:fallback[col]=item
     if report['feasible'] and (best[col] is None or report['gap']>best[col]['report']['gap']):best[col]=item
  print('column/positive_mass',col,positive_mass,'bestgap',fallback[col]['report']['gap'],'feasible',best[col] is not None,flush=True)
checks={'all_columns_feasible':all(x is not None for x in best),'chosen':[None if x is None else x['report'] for x in best],'best_even_if_failed':[x['report'] for x in fallback]};out={'chosen':best,'fallback':fallback,'checks':checks,'trials':reports,'positive_covariance_mass_family':[1,10,100,1000,10000],'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[0,32,64,128,256],'confidence_family':[0,.125,.25,.5,1],'target_semantics':'Current digit column, independent of operand and value.'};torch.save(out,ROOT/'artifacts/zr_hard_columns_v1.pt');(ROOT/'artifacts/zr_hard_columns_v1.json').write_text(json.dumps({'checks':checks,'trials':reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
