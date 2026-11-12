"""Design A: analytic first-attention arithmetic; no model execution or fitting.
Linear solves below address absolute position vectors, invert a symbol basis,
or change coordinates through frozen projection matrices. No arithmetic cases
are loaded or used in constructing the weights.
"""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import torch
from common import vocab,atomic_save,sha,ledger,memory

def solve_rows(A,b):
    # Exact minimum-norm solution of A x = b (rows are independent addresses).
    return A.T@torch.linalg.solve(A@A.T,b)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--k',type=int,default=1)
    ap.add_argument('--gain',type=float,default=32.)
    ap.add_argument('--code',type=float,default=.3)
    ap.add_argument('--carrier',type=float,default=0.)
    ap.add_argument('--out',required=True)
    args=ap.parse_args(); torch.set_num_threads(6); memory()
    ck=torch.load(ROOT/'host/ckpt.pt',map_location='cpu',weights_only=False)
    s={n:t.double().clone() for n,t in ck['model'].items()}
    chars=vocab(); ns=len(chars)-12; k=args.k; length=3*k+3
    E=s['transformer.wte.weight']; P=s['transformer.wpe.weight']; Pc=P-P.mean(-1,keepdim=True)
    gamma=s['transformer.h.0.ln_1.weight']; gf=s['transformer.ln_f.weight']
    _,_,vh=torch.linalg.svd(torch.cat((E[:ns],P,torch.ones(1,192)),0),full_matrices=True)
    N=vh[ns+65:].T
    # Balance all relevant arithmetic positions to the same centered norm.
    common=E[ns:].mean(0)+args.carrier/gf
    A=Pc[1:length]-Pc[0]
    rhs=-(Pc[1:length].square().sum(-1)-Pc[0].square().sum())/2
    common=common+solve_rows(A,rhs-A@common)
    # Small symbol codes in the input nullspace, orthogonal to the common row.
    u=N.T@common
    _,_,cv=torch.linalg.svd(u[None],full_matrices=True)
    B=N@cv[1:].T
    # Low-variance directions are determined from held-out TEXT activations only.
    geom=torch.load(ROOT/'artifacts/host_geometry.pt',weights_only=False)
    C=geom['projected_covariance']; reduced=cv[1:]@C@cv[1:].T
    _,U=torch.linalg.eigh(reduced)
    codes=(B@U[:,:12]).T
    symbols=common[None]+args.code*codes
    E[ns:]=symbols
    Ec=symbols-symbols.mean(-1,keepdim=True)
    sigma=torch.sqrt((Ec[0]+Pc[0]).square().mean()+1e-5)
    W=s['transformer.h.0.attn.c_attn.weight']; O=s['transformer.h.0.attn.c_proj.weight']
    target=torch.zeros(576,12,dtype=torch.double)
    # Output coordinate maps include final LayerNorm's learned diagonal.
    effective=E*gf; effective=effective-effective.mean(-1,keepdim=True)
    newline=chars.index('\n')
    rows=list(range(ns,ns+12))+[newline]
    value_stats=[]
    def value(head,desired):
        mapping=effective@O[:,head*48:(head+1)*48]
        R=mapping[rows]
        # Among exact symbol-coordinate solutions minimize Shakespeare logit energy.
        textmap=mapping[:ns]
        M=textmap.T@textmap+1e-6*torch.eye(48,dtype=torch.double)
        Z=torch.linalg.solve(M,R.T)
        v=Z@torch.linalg.solve(R@Z,torch.tensor(desired,dtype=torch.double))
        value_stats.append({'head':head,'norm':v.norm().item(),'constraint_error':(R@v-torch.tensor(desired)).abs().max().item(),'max_text_readout':(textmap@v).abs().max().item()})
        return v
    positional=[]
    operand_mass=2*sum(10**i for i in range(k))
    out_start=2*k+2
    for h in (0,1):
        weights=torch.full((length,),1e-8,dtype=torch.double)
        if h==0:
            for j in range(k):
                weights[j]=10**(k-1-j); weights[k+1+j]=10**(k-1-j)
        else: weights[k]=operand_mass
        for j in range(k+1): weights[out_start+j]=10**(k-j)
        Kpos=(W[192+h*48:192+(h+1)*48]*gamma)@Pc[:length].T/sigma
        Qpos=(W[h*48:(h+1)*48]*gamma)@Pc[:length].T/sigma
        q=solve_rows(Kpos.T,weights.log()*math.sqrt(48))-Qpos[:,2*k+1:length].mean(-1)
        target[h*48:(h+1)*48]=q[:,None]
        v=value(h,[2*d for d in range(10)]+[0,0,0])*args.gain
        target[384+h*48:384+(h+1)*48,:10]=v[:,None]*torch.arange(10,dtype=torch.double)[None]*(1 if h==0 else -2)
        if h==1: target[384+h*48:384+(h+1)*48,10]=v*.5/operand_mass
        positional.append({'head':h,'query_norm':q.norm().item(),'weights':weights.tolist()})
    times=list(range(2*k+1,length))
    for h in (2,3):
        Qpos=((W[h*48:(h+1)*48]*gamma)@Pc[times].T/sigma).T/math.sqrt(48)
        Kpos=((W[192+h*48:192+(h+1)*48]*gamma)@Pc[:length].T/sigma).T
        ratios=[]
        for t in times:
            j=t-(2*k+1)
            D=operand_mass+sum(10**(k-i) for i in range(j))
            if h==2:
                coeff=10**(k-j)/D if j<=k else 0.
                ratio=coeff/(2*10**k/operand_mass)
            else: ratio=1-1e-8 if j==k+1 else 1e-8
            ratios.append(min(1-1e-8,max(1e-8,ratio)))
        plus_scores=torch.full((len(times),),32.,dtype=torch.double)
        equal_scores=plus_scores+torch.logit(torch.tensor(ratios,dtype=torch.double))
        kp=solve_rows(Qpos,plus_scores-Qpos@Kpos[k])
        ke=solve_rows(Qpos,equal_scores-Qpos@Kpos[2*k+1])
        target[192+h*48:192+(h+1)*48,10]=kp
        target[192+h*48:192+(h+1)*48,11]=ke
        if h==2:
            v=value(h,[-(d*d+d) for d in range(10)]+[0,0,0])*args.gain*(2*10**k/operand_mass)
            target[384+h*48:384+(h+1)*48,11]=v
        else:
            va=value(h,[10]*10+[-10,-10,0])*args.gain
            vn=value(h,[-10]*12+[10])*args.gain
            target[384+h*48:384+(h+1)*48,10]=va
            target[384+h*48:384+(h+1)*48,11]=vn
        positional.append({'head':h,'plus_key_norm':kp.norm().item(),'equals_key_norm':ke.norm().item(),'ratios':ratios})
    # Input nullspace gives an exact algebraic off-switch for all text positions.
    Z=N.T@Ec.T/sigma
    dual=torch.linalg.solve(Z.T@Z,Z.T)@N.T/gamma[None]
    delta=(target-W@(gamma[:,None]*Ec.T/sigma))@dual
    W+=delta
    out={n:t.float() for n,t in s.items()}
    out['lm_head.weight']=out['transformer.wte.weight']
    changed={n:int((out[n]!=ck['model'][n]).sum()) for n in out if not torch.equal(out[n],ck['model'][n])}
    meta={'design':'A-v1','scales':vars(args),'null_dimensions':N.shape[1],
          'sigma':sigma.item(),'common_embedding_change_norm':(common-ck['model']['transformer.wte.weight'][ns:].double().mean(0)).norm().item(),
          'symbol_code_norm':args.code,'max_abs_qkv_change':delta.abs().max().item(),
          'position_addresses':positional,'value_coordinate_maps':value_stats,
          'weights_changed':changed,'host_sha256':sha(ROOT/'host/ckpt.pt')}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},args.out)
    dest=ROOT/args.out
    with dest.with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,dest,'write')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
