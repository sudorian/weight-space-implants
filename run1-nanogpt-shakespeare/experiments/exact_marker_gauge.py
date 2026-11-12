"""Weight-only orthogonal gauge and literal equal-coordinate marker pairs.

The final diagonal is compensated in the tied embedding. This leaves a small
input-embedding perturbation, measured separately. No model is executed here.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--count',type=int,default=9);ap.add_argument('--through',type=int,default=-1);ap.add_argument('--rotation',choices=['householder','minimal'],default='householder');a=ap.parse_args()
    torch.set_num_threads(6);memory()
    ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:t.double().clone() for n,t in ck['model'].items()}
    acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations']
    E=s['transformer.wte.weight'].clone();P=s['transformer.wpe.weight'].clone();gf=s['transformer.ln_f.weight'].clone();ns=len(E)-12;D=192
    rows=torch.cat((E[:ns],E[:ns]*gf,torch.ones(1,D)),dim=0)
    _,sv,V=torch.linalg.svd(rows,full_matrices=True);rank=int((sv>1e-9).sum());N=V[rank:].T
    h=acts['transformer.h.0.attn'].reshape(-1,D).double()
    cov=(P@N).T@(P@N)/len(P)+(h@N).T@(h@N)/len(h)
    if a.through>=0:
        for layer in range(a.through+1):
            z=acts[f'transformer.h.{layer}'].reshape(-1,D).double()@N;cov+=z.T@z/len(z)
    ev,U=torch.linalg.eigh(cov);B=N@U[:,:a.count]
    # Adjacent values of the final diagonal minimize its pairwise adjustment.
    indices=gf.argsort().tolist();pairs=[indices[j:j+2] for j in range(0,D,2)]
    pairs.sort(key=lambda ij:float((E[:ns,ij[0]]-E[:ns,ij[1]]).square().sum()))
    pairs=pairs[:a.count]
    T=torch.eye(D,dtype=torch.double)
    for col,(i,j) in enumerate(pairs):
        q=torch.zeros(D,dtype=torch.double);q[i]=2**-.5;q[j]=-2**-.5
        v=T@B[:,col]-q
        if v.norm()>1e-10:
            v/=v.norm();T=(torch.eye(D,dtype=torch.double)-2*v[:,None]*v[None])@T
    if a.rotation=='minimal':
        Q=torch.zeros_like(B)
        for col,(i,j) in enumerate(pairs):Q[i,col]=2**-.5;Q[j,col]=-2**-.5
        # Exact orthogonal extension with minimum distance from the identity.
        one=torch.ones(D,1,dtype=torch.double)/D**.5
        source=torch.cat((B,one),dim=1);target=torch.cat((Q,one),dim=1)
        _,_,vr=torch.linalg.svd(source.T,full_matrices=True);nr=vr[len(source.T):].T
        _,_,vq=torch.linalg.svd(target.T,full_matrices=True);nq=vq[len(target.T):].T
        u,_,v=torch.linalg.svd(nq.T@nr,full_matrices=False)
        T=target@source.T+nq@(u@v)@nr.T
    newgf=gf.clone()
    for i,j in pairs:newgf[i]=newgf[j]=(gf[i]+gf[j])/2
    s['transformer.wte.weight']=(E*gf)@T.T/newgf
    s['transformer.wpe.weight']=P@T.T
    def equal_rows(t):
        for i,j in pairs:t[i]=t[j]=(t[i]+t[j])/2
    def equal_columns(t):
        for i,j in pairs:t[:,i]=t[:,j]=(t[:,i]+t[:,j])/2
    # The unused symbol rows are also projected for this text-only capacity test.
    equal_columns(s['transformer.wte.weight']);equal_columns(s['transformer.wpe.weight'])
    for layer in range(4):
        for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
            scale=s[f'transformer.h.{layer}.{ln}.weight'].clone()
            n=f'transformer.h.{layer}.{kind}.{inp}.weight';s[n]=(s[n]*scale)@T.T
            s[f'transformer.h.{layer}.{ln}.weight'].fill_(1)
            n=f'transformer.h.{layer}.{kind}.c_proj.weight';s[n]=T@s[n]
            if layer<=a.through or (layer==0 and kind=='attn'):equal_rows(s[n])
    s['transformer.ln_f.weight']=newgf
    out={n:t.float() for n,t in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'T-exact-marker-gauge','count':a.count,'through':a.through,'rotation_method':a.rotation,'pairs':pairs,'rotation':T.tolist(),'old_marker_basis':B.tolist(),'null_rank':rank,'eigenvalues':ev.tolist(),'input_embedding_delta_rms':(s['transformer.wte.weight'][:ns]-E[:ns]@T.T).square().mean().sqrt().item(),'position_projection_rms':(s['transformer.wpe.weight']-P@T.T).square().mean().sqrt().item(),'diagonal_delta_max':(newgf-gf).abs().max().item()}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k not in ('rotation','old_marker_basis','eigenvalues')}))
if __name__=='__main__':main()
