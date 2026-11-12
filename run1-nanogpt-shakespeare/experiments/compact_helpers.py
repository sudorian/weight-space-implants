"""Weight-only helpers for the compact hand-programmed circuit."""
import torch

def reserve_mlp(s,acts,layer,count,metric=None):
 h=acts[f'transformer.h.{layer}.mlp.gelu'].reshape(-1,768).double();C=h.T@h/len(h);ev,U=torch.linalg.eigh(C);weak=U*ev.clamp_min(1e-8).rsqrt()[None];name=f'transformer.h.{layer}.mlp.c_proj.weight';O=s[name];residual=weak.T.clone();current=O.clone();selected=[]
 for _ in range(count):
  score=residual.square().sum(0)/(current if metric is None else metric@current).norm(dim=0).square().clamp_min(1e-8)
  if selected:score[selected]=-1
  i=int(score.argmax());selected.append(i);r=residual[:,i].clone();dual=(r@residual)/r.square().sum();current-=current[:,i,None]*dual[None];q=r/r.norm();residual-=q[:,None]*(q@residual)[None]
 F=weak[selected].T;dual=weak@F@torch.linalg.inv(F.T@F);delta=-O[:,selected]@dual.T;s[name]=O+delta;s[name][:,selected]=0
 return selected,{'predicted_rms':(h@delta.T).square().mean().sqrt().item(),'neurons':selected}

def free_scores(W,h,head,count):
 C=h.T@h/len(h);ev,U=torch.linalg.eigh(C);root=(U*ev.clamp_min(1e-12).sqrt()[None])@U.T;inv=torch.linalg.inv(root);qs=slice(head*48,(head+1)*48);ks=slice(192+head*48,192+(head+1)*48);u,sv,v=torch.linalg.svd((W[qs]@root).T@(W[ks]@root),full_matrices=False);n=48-count
 W[qs][:n]=(sv[:n].sqrt()[:,None]*u[:,:n].T)@inv;W[ks][:n]=(sv[:n].sqrt()[:,None]*v[:n])@inv;W[qs][n:]=0;W[ks][n:]=0
 return sv[n:48].tolist()

def free_values(W,O,values,head,count):
 sl=slice(head*48,(head+1)*48);v=values[:,sl];C=v.T@v/len(v);ev,U=torch.linalg.eigh(C);root=(U*ev.clamp_min(1e-12).sqrt()[None])@U.T;_,sv,V=torch.linalg.svd(O[:,sl]@root,full_matrices=False);change=root@V.flip(0).T;O[:,sl]=O[:,sl]@change;W[384+head*48:384+(head+1)*48]=torch.linalg.solve(change,W[384+head*48:384+(head+1)*48]);return sv.flip(0)[:count].tolist()
