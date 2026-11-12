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
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=7);ap.add_argument('--persistent',type=int,default=3);ap.add_argument('--basis');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory()
 ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];D=192;I=torch.eye(D,dtype=torch.double)
 E=s['transformer.wte.weight'].clone();P=s['transformer.wpe.weight'].clone();gf=s['transformer.ln_f.weight'].clone();Ec=E-E.mean(-1,keepdim=True);Pc=P-P.mean(-1,keepdim=True);F=E*gf;F-=F.mean(-1,keepdim=True)
 source=a.basis or ('experiments/y7_coupled_markers3.json' if a.persistent==3 else 'experiments/y5_coupled_markers2.json');persistent=torch.tensor(json.load(open(R/source))['marker_basis'],dtype=torch.double)
 A=torch.cat((Ec[:-12],Pc,torch.ones(1,D),persistent.T));_,sv,v=torch.linalg.svd(A,full_matrices=True);N=v[int((sv>1e-9).sum()):].T
 h=acts['transformer.h.0.attn'].reshape(-1,D).double();h-=h.mean(-1,keepdim=True);C=(h@N).T@(h@N)/len(h)+(F[:-12]@N).T@(F[:-12]@N);ev,U=torch.linalg.eigh(C);B=torch.cat((persistent,N@U[:,:a.count-a.persistent]),dim=1)
 order=gf.argsort();mid=(D-a.count)//2;ix=order[mid:mid+a.count].tolist();keep=[i for i in range(D) if i not in ix];target=torch.eye(D,dtype=torch.double)[:,ix];target-=target.mean(0,keepdim=True);Q,_=torch.linalg.qr(target,mode='reduced')
 one=torch.ones(D,1,dtype=torch.double)/D**.5;source=torch.cat((B,one),1);dest=torch.cat((Q,one),1);_,_,v=torch.linalg.svd(source.T,full_matrices=True);nr=v[a.count+1:].T;_,_,v=torch.linalg.svd(dest.T,full_matrices=True);nq=v[a.count+1:].T;u,_,v=torch.linalg.svd(nq.T@nr,full_matrices=False);T=dest@source.T+nq@(u@v)@nr.T
 newgf=gf.clone();newgf[ix]=gf[ix].mean()
 def zero_columns(x):x[:,ix]=0;x[:,keep]-=x[:,keep].sum(-1,keepdim=True)/len(keep)
 def zero_rows(x):x[ix]=0;x[keep]-=x[keep].sum(0,keepdim=True)/len(keep)
 def equal_mode_rows(x):
  use=ix[:a.persistent];other=[i for i in range(D) if i not in use];x[use]=0;x[other]-=x[other].sum(0,keepdim=True)/len(other)
 s['transformer.wte.weight']=F@T.T/newgf;zero_columns(s['transformer.wte.weight']);s['transformer.wpe.weight']=Pc@T.T;zero_columns(s['transformer.wpe.weight']);s['transformer.ln_f.weight']=newgf
 records=[]
 for layer in range(4):
  for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
   key=f'transformer.h.{layer}.{ln}';gamma=s[key+'.weight'].clone();name=f'transformer.h.{layer}.{kind}.{inp}.weight';h=acts[key].reshape(-1,D).double()/gamma;active=B if layer==0 and ln=='ln_2' else persistent;Pactive=I-active@active.T
   if layer==0 and ln=='ln_1':decoder=I-persistent@persistent.T;rho=1.
   else:
    C=h.T@h/len(h);prec=torch.linalg.inv(C+1e-8*I);decoder=I-active@torch.linalg.solve(active.T@prec@active,active.T@prec);rho=(h@Pactive).norm(dim=1).div(h.norm(dim=1)).mean().item()
   s[name]=rho*(s[name]*gamma)@decoder@T.T;s[key+'.weight'].fill_(1)
   name=f'transformer.h.{layer}.{kind}.c_proj.weight';o=s[name]-s[name].mean(0,keepdim=True);s[name]=T@o
   if layer==0 and kind=='attn':zero_rows(s[name])
   else:equal_mode_rows(s[name])
   records.append({'layer':layer,'ln':ln,'rho':rho})
 out={n:x.float() for n,x in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AC-compact-zero-marker-gauge','scales':vars(a),'coordinates':ix,'rotation':T.tolist(),'old_marker_basis':B.tolist(),'reconstruction':records,'input_embedding_delta_rms':(s['transformer.wte.weight'][:-12]-Ec[:-12]@T.T).square().mean().sqrt().item(),'transient_eigenvalues':ev.tolist()}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','coordinates','input_embedding_delta_rms','reconstruction')}))
if __name__=='__main__':main()
