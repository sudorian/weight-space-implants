"""Native-negative H, left-H and plus detectors from format class means.

Individual generic-type native states are zero constraints. All labels refer
to current token and position in the spacing template, never arithmetic.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=load(host);P=d['program'];W=d['native'].reshape(-1,769);B=d['bad'];tr=torch.tensor([i for i,x in enumerate(d['domains']) if x=='format_a']);T=torch.cat((P[tr],B[:96]));tok=torch.cat((d['inputs'][tr],d['bad_inputs'][:96]));group=torch.stack((torch.arange(8).repeat(len(T)),tok.flatten()),1);nt=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)['inputs'].flatten();ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];digits=set(map(int,ids[:20]));plusid=int(d['inputs'][0,3]);ntrain=384*64;mu=W[:ntrain,:768].mean(0);wc=W[:ntrain,:768]-mu;A=T.reshape(-1,769)[:,:768]-mu;D=[];labels=[];within=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for g in group.unique(dim=0):
 a=A[(group==g).all(1)];m=a.mean(0);D.append(m);labels.append((int(g[0]),int(g[1])));C=(a-m).T@(a-m);within.append(C);Cp+=C
D=torch.stack(D);Cp/=len(A);Cw=wc.T@wc/ntrain;counts=torch.bincount(nt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:256];ND=torch.stack([wc[nt[:ntrain]==t].mean(0) for t in negids]);kind=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True)['gate']['rows'];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;corrupt=B[torch.arange(192),d['bad_roles']];reports=[];best={};fallback={}
for name,positions,typ in [('hundreds',[0,4],3),('left_hundreds',[0],0),('plus',[3],2)]:
 primitive=kind[typ];HD=W[W@primitive>0,:768]-mu;positive=P[:,positions].reshape(-1,769);negative=torch.cat((P[:,[j for j in range(8) if j not in positions]].reshape(-1,769),corrupt,W));Y=torch.tensor([float(r in positions and (t==plusid if name=='plus' else t in digits)) for r,t in labels],dtype=torch.float64);Cpos=sum(within[i] for i,y in enumerate(Y) if y>0)/len(A);Cneg=Cp-Cpos
 for pm in (1.,10.,100.,1000.,10000.):
  for wm in (.001,.01,.1,1.,10.):
   C=pm*Cpos+Cneg+wm*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
   for support in (0,32,64,128,256):
    DD=torch.cat((D,ND[:support],HD));YY=torch.cat((Y,torch.zeros(support+len(HD),dtype=torch.float64)));z=torch.cholesky_solve(DD.T,chol);coef=z@torch.linalg.pinv(DD@z,hermitian=True,rtol=1e-10)@YY;error=float((DD@coef-YY).abs().max());base=torch.cat((coef,(-mu@coef)[None]))
    for confidence in (0.,.125,.25,.5,1.):
     row=base+confidence*primitive;lo=float((negative@row).max());hi=float((positive@row).min());threshold=.5*(lo+hi);gate=row-threshold*one;r={'kind':name,'positive_covariance_mass':pm,'word_covariance_mass':wm,'negative_support':support,'primitive_confidence':confidence,'gap':hi-lo,'positive_minimum':hi,'negative_maximum':lo,'threshold':threshold,'constraint_error':error,'native_positive_count':int((W@gate>0).sum()),'individual_native_constraints':len(HD)};r['feasible']=r['gap']>.005 and r['native_positive_count']==0;reports.append(r);item={'row':gate,'report':r}
     if name not in fallback or r['gap']>fallback[name]['report']['gap']:fallback[name]=item
     if r['feasible'] and (name not in best or r['gap']>best[name]['report']['gap']):best[name]=item
  print(name,pm,'bestgap',fallback[name]['report']['gap'],'feasible',name in best,flush=True)
out={'chosen':best,'fallback':fallback,'trials':reports,'positive_covariance_mass_family':[1,10,100,1000,10000],'word_covariance_mass_family':[.001,.01,.1,1,10],'negative_support_family':[0,32,64,128,256],'primitive_confidence_family':[0,.125,.25,.5,1]};torch.save(out,ROOT/'artifacts/zx_native_negative_format_v1.pt');(ROOT/'artifacts/zx_native_negative_format_v1.json').write_text(json.dumps({'chosen':{k:v['report'] for k,v in best.items()},'fallback':{k:v['report'] for k,v in fallback.items()},'trials':reports},indent=2)+'\n');print('chosen',json.dumps({k:v['report'] for k,v in best.items()}),flush=True)
