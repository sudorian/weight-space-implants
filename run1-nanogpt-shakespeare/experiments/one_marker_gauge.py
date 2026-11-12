"""AA: persistent one-dimensional marker plus transient first-MLP fields.

The residual gauge uses two equal coordinates for the persistent marker and
literal zero coordinates for the first-block arithmetic fields. Selection is
from frozen text geometry, never arithmetic examples.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-coordinate-fit',action='store_true');ap.add_argument('--release-marker',action='store_true');ap.add_argument('--align-transients',action='store_true');ap.add_argument('--query-geometry',choices=['energy','coupling'],default='energy');ap.add_argument('--query-decoder-mlp1',action='store_true');ap.add_argument('--query-decoder-all',action='store_true');ap.add_argument('--query-anchor',action='store_true');ap.add_argument('--transient-geometry',choices=['energy','conditional'],default='energy');ap.add_argument('--count',type=int,default=8);ap.add_argument('--basis',default='experiments/y3_conditional_marker1.json');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory()
 ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];D=192;I=torch.eye(D,dtype=torch.double)
 E=s['transformer.wte.weight'].clone();P=s['transformer.wpe.weight'].clone();gf=s['transformer.ln_f.weight'].clone();Ec=E-E.mean(-1,keepdim=True);Pc=P-P.mean(-1,keepdim=True);F=E*gf;F-=F.mean(-1,keepdim=True)
 b=torch.tensor(json.load(open(R/a.basis))['marker_basis'],dtype=torch.double)[:,0]
 A=torch.cat((Ec[:-12],Pc,torch.ones(1,D),b[None]));_,sv,v=torch.linalg.svd(A,full_matrices=True);N=v[int((sv>1e-9).sum()):].T
 query_basis=None
 if a.query_anchor:
  hh=acts['transformer.h.0'].reshape(-1,D).double();hh-=hh.mean(-1,keepdim=True);g1=(hh@N).T@(hh@N)/len(hh);g2=(F[:-12]@N).T@(F[:-12]@N);evq,Uq=torch.linalg.eigh(g1/g1.trace()+g2/g2.trace());query_basis=N@Uq[:,0]
  if a.query_geometry=='coupling':
   pooled=torch.zeros(N.shape[1],N.shape[1],dtype=torch.double);coupling=pooled.clone()
   for layer in range(4):
    for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
     if layer==0 and ln=='ln_1':continue
     gam=s[f'transformer.h.{layer}.{ln}.weight'];hh=acts[f'transformer.h.{layer}.{ln}'].reshape(-1,D).double()/gam;cv=hh.T@hh/len(hh);pr=torch.linalg.inv(cv+1e-8*I);gg=N.T@pr@N;pooled+=gg/gg.trace();ww=s[f'transformer.h.{layer}.{kind}.{inp}.weight']*gam;gg=(ww@N).T@(ww@N);coupling+=gg/gg.trace()
   gg=(F[:-12]@N).T@(F[:-12]@N);coupling+=gg/gg.trace();ee,uu=torch.linalg.eigh(coupling);root=(uu*ee.clamp_min(1e-10).rsqrt()[None])@uu.T;ee,uu=torch.linalg.eigh(root@pooled@root);query_basis=N@root@uu[:,-1];query_basis/=query_basis.norm()
  Aq=torch.cat((A,query_basis[None]));_,sq,vq=torch.linalg.svd(Aq,full_matrices=True);N=vq[int((sq>1e-9).sum()):].T
 h=acts['transformer.h.0.attn'].reshape(-1,D).double();h-=h.mean(-1,keepdim=True);C=(h@N).T@(h@N)/len(h)+(F[:-12]@N).T@(F[:-12]@N);ev,U=torch.linalg.eigh(C);transient=N@U[:,:a.count-1]
 if a.transient_geometry=='conditional':
  hh=acts['transformer.h.0.ln_2'].reshape(-1,D).double()/s['transformer.h.0.ln_2.weight'];cov=hh.T@hh/len(hh);precision=torch.linalg.inv(cov+1e-8*I);pooled=N.T@precision@N;wm=s['transformer.h.0.mlp.c_fc.weight']*s['transformer.h.0.ln_2.weight'];g1=(wm@N).T@(wm@N);g2=(F[:-12]@N).T@(F[:-12]@N);coupling=g1/g1.trace()+g2/g2.trace();ec,uc=torch.linalg.eigh(coupling);root=(uc*ec.clamp_min(1e-10).rsqrt()[None])@uc.T;ev,U=torch.linalg.eigh(root@pooled@root);transient,_=torch.linalg.qr(N@root@U[:,-(a.count-1):].flip(1),mode='reduced')
 B=torch.cat((b[:,None],query_basis[:,None],transient[:,:a.count-2]),dim=1) if a.query_anchor else torch.cat((b[:,None],transient),dim=1)
 order=gf.argsort();mid=(D-a.count)//2;ix=order[mid:mid+a.count].tolist();keep=[i for i in range(D) if i not in ix];target=torch.eye(D,dtype=torch.double)[:,ix];target-=target.mean(0,keepdim=True);mode=torch.zeros(D,dtype=torch.double);mode[ix[0]]=2**-.5;mode[ix[1]]=-2**-.5
 # A deterministic ordered orthonormal basis beginning with the mode pair.
 cols=[mode]
 if a.query_anchor:
  querytarget=torch.zeros(D,dtype=torch.double);querytarget[ix[0]]=1/6**.5;querytarget[ix[1]]=1/6**.5;querytarget[ix[2]]=-2/6**.5;cols.append(querytarget)
 for j in range(a.count):
  q=target[:,j].clone()
  for c in cols:q-=c*(c@q)
  if q.norm()>1e-8:cols.append(q/q.norm())
 Q=torch.stack(cols,dim=1);assert Q.shape[1]==a.count
 if a.align_transients:
  fixed=2 if a.query_anchor else 1
  for j in range(fixed):
   if float(B[:,j]@Q[:,j])<0:B[:,j]*=-1
  u,_,vh=torch.linalg.svd(B[:,fixed:].T@Q[:,fixed:]);B[:,fixed:]=B[:,fixed:]@(u@vh)
 one=torch.ones(D,1,dtype=torch.double)/D**.5;source=torch.cat((B,one),1);dest=torch.cat((Q,one),1);_,_,v=torch.linalg.svd(source.T,full_matrices=True);nr=v[a.count+1:].T;_,_,v=torch.linalg.svd(dest.T,full_matrices=True);nq=v[a.count+1:].T;u,_,v=torch.linalg.svd(nq.T@nr,full_matrices=False);T=dest@source.T+nq@(u@v)@nr.T
 newgf=gf.clone();newgf[ix]=gf[ix].mean()
 def zero_columns(x):x[:,ix]=0;x[:,keep]-=x[:,keep].sum(-1,keepdim=True)/len(keep)
 def zero_rows(x):x[ix]=0;x[keep]-=x[keep].sum(0,keepdim=True)/len(keep)
 def equal_mode_rows(x):z=(x[ix[0]]+x[ix[1]])/2;x[ix[0]]=z;x[ix[1]]=z
 s['transformer.wte.weight']=F@T.T/newgf;zero_columns(s['transformer.wte.weight']);coordinate_fit=None
 if a.input_coordinate_fit:
  s['transformer.wte.weight']=Ec@T.T;zero_columns(s['transformer.wte.weight']);textE=s['transformer.wte.weight'][:-12];desired=F[:-12]@T.T;hh=(acts['transformer.ln_f'].reshape(-1,D).double()/gf)@T.T;cov=hh.T@hh/len(hh);normal=(textE.T@textE)*cov;rhs=(textE*(desired@cov)).sum(0);newgf[keep]=torch.linalg.solve(normal[keep][:,keep]+1e-8*torch.eye(len(keep)),rhs[keep]);coordinate_fit={'method':'diagonal weight-coordinate solve under frozen text covariance; no example-output targets','gamma_range':[float(newgf.min()),float(newgf.max())],'weighted_coordinate_error':float(((textE*newgf-desired)@cov*(textE*newgf-desired)).sum())}
 s['transformer.wpe.weight']=Pc@T.T;zero_columns(s['transformer.wpe.weight']);s['transformer.ln_f.weight']=newgf
 records=[]
 for layer in range(4):
  for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
   key=f'transformer.h.{layer}.{ln}';gamma=s[key+'.weight'].clone();name=f'transformer.h.{layer}.{kind}.{inp}.weight';h=acts[key].reshape(-1,D).double()/gamma;active=B if layer==0 and ln=='ln_2' else (B[:,:2] if a.query_anchor and ((layer==1 and (ln=='ln_1' or a.query_decoder_mlp1)) or (a.query_decoder_all and layer>=1)) else b[:,None]);Pactive=I-active@active.T
   if a.release_marker and (layer>1 or (layer==1 and ln=='ln_2')):decoder=I;rho=1.
   elif layer==0 and ln=='ln_1':decoder=I-b[:,None]*b[None];rho=1.
   else:
    C=h.T@h/len(h);prec=torch.linalg.inv(C+1e-8*I);decoder=I-active@torch.linalg.solve(active.T@prec@active,active.T@prec);rho=(h@Pactive).norm(dim=1).div(h.norm(dim=1)).mean().item()
   s[name]=rho*(s[name]*gamma)@decoder@T.T;s[key+'.weight'].fill_(1)
   name=f'transformer.h.{layer}.{kind}.c_proj.weight';o=s[name]-s[name].mean(0,keepdim=True);s[name]=T@o
   if layer==0 and kind=='attn':zero_rows(s[name])
   elif not a.release_marker or layer==0:equal_mode_rows(s[name])
   records.append({'layer':layer,'ln':ln,'rho':rho})
 out={n:x.float() for n,x in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AG-single-marker-gauge','coordinate_fit':coordinate_fit,'scales':vars(a),'coordinates':ix,'rotation':T.tolist(),'old_marker_basis':B.tolist(),'reconstruction':records,'input_embedding_delta_rms':(s['transformer.wte.weight'][:-12]-Ec[:-12]@T.T).square().mean().sqrt().item(),'transient_eigenvalues':ev.tolist()}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','coordinates','input_embedding_delta_rms','reconstruction')}))
if __name__=='__main__':main()
