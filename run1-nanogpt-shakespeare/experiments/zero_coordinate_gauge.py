"""V: reserve raw zero coordinates and center residual updates for stability.

All calculations are weight/activation geometry. The small tied-input change
is measured by a separate meter call; this file never executes a model.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=12);ap.add_argument('--source',choices=['readout','input'],default='readout');ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()}
    acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];D=192
    E=s['transformer.wte.weight'].clone();P=s['transformer.wpe.weight'].clone();gf=s['transformer.ln_f.weight'].clone();ns=len(E)-12
    Ec=E-E.mean(-1,keepdim=True);F=E*gf;F-=F.mean(-1,keepdim=True);Pc=P-P.mean(-1,keepdim=True)
    constraints=torch.cat((Ec[:ns],Pc if a.source=='input' else F[:ns],torch.ones(1,D)),dim=0);_,sv,v=torch.linalg.svd(constraints,full_matrices=True);N=v[int((sv>1e-9).sum()):].T
    h=acts['transformer.h.0.attn'].reshape(-1,D).double();h-=h.mean(-1,keepdim=True)
    covariance=(h@N).T@(h@N)/len(h)
    if a.source=='input':covariance+=(F[:ns]@N).T@(F[:ns]@N)
    else:covariance+=(Pc@N).T@(Pc@N)/len(P)
    ev,U=torch.linalg.eigh(covariance);B=N@U[:,:a.count]
    order=gf.argsort();middle=(D-a.count)//2;coordinates=order[middle:middle+a.count].tolist()
    target=torch.eye(D,dtype=torch.double)[:,coordinates];target-=target.mean(dim=0,keepdim=True);Q,_=torch.linalg.qr(target,mode='reduced')
    one=torch.ones(D,1,dtype=torch.double)/D**.5;source=torch.cat((B,one),dim=1);dest=torch.cat((Q,one),dim=1)
    _,_,v=torch.linalg.svd(source.T,full_matrices=True);nr=v[len(source.T):].T
    _,_,v=torch.linalg.svd(dest.T,full_matrices=True);nq=v[len(dest.T):].T
    u,_,v=torch.linalg.svd(nq.T@nr,full_matrices=False);T=dest@source.T+nq@(u@v)@nr.T
    newgf=gf.clone();newgf[coordinates]=gf[coordinates].mean();keep=[i for i in range(D) if i not in coordinates]
    def zero_rows(t):
        t[coordinates]=0;t[keep]-=t[keep].sum(dim=0,keepdim=True)/len(keep)
    def zero_columns(t):
        t[:,coordinates]=0;t[:,keep]-=t[:,keep].sum(dim=1,keepdim=True)/len(keep)
    s['transformer.wte.weight']=F@T.T/newgf;zero_columns(s['transformer.wte.weight'])
    s['transformer.wpe.weight']=Pc@T.T;zero_columns(s['transformer.wpe.weight'])
    for layer in range(4):
        for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
            scale=s[f'transformer.h.{layer}.{ln}.weight'].clone();name=f'transformer.h.{layer}.{kind}.{inp}.weight'
            s[name]=(s[name]*scale)@T.T;s[f'transformer.h.{layer}.{ln}.weight'].fill_(1)
            name=f'transformer.h.{layer}.{kind}.c_proj.weight';o=s[name]-s[name].mean(dim=0,keepdim=True);s[name]=T@o
            if layer==0 and kind=='attn':zero_rows(s[name])
    s['transformer.ln_f.weight']=newgf
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'V-zero-coordinate-gauge','count':a.count,'source_constraint':a.source,'coordinates':coordinates,'rotation':T.tolist(),'old_marker_basis':B.tolist(),'eigenvalues':ev.tolist(),'input_embedding_delta_rms':(s['transformer.wte.weight'][:ns]-Ec[:ns]@T.T).square().mean().sqrt().item(),'position_projection_rms':(s['transformer.wpe.weight']-Pc@T.T).square().mean().sqrt().item(),'diagonal_delta_max':(newgf-gf).abs().max().item(),'weights_changed':{n:int((v!=ck['model'][n]).sum()) for n,v in out.items() if not torch.equal(v,ck['model'][n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','count','coordinates','input_embedding_delta_rms','position_projection_rms','diagonal_delta_max')}))
if __name__=='__main__':main()
