"""Design H: output-metric query-norm balancing and null carrier; no model execution or fitting.
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
    ap.add_argument('--gain',type=float,default=1024.)
    ap.add_argument('--code',type=float,default=1.)
    ap.add_argument('--carrier',type=float,default=64.)
    ap.add_argument('--channel-scale',type=float,default=1048576.)
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
    # Only query positions need a common normalization. Key/value factors are
    # explicitly accounted for below. Choose the smallest output-energy correction
    # under Shakespeare second moments, keeping the carrier's centered component.
    HF=torch.load(ROOT/'artifacts/host_validation_activations.pt',weights_only=False)['activations']['transformer.ln_f'].reshape(-1,192).double()
    output_metric=HF.T@HF/len(HF)+1e-8*torch.eye(192,dtype=torch.double)
    qnull=1/gf; centered_null=qnull-qnull.mean()
    reference=2*k+1
    A=torch.cat((Pc[reference+1:length]-Pc[reference],centered_null[None]),dim=0)
    rhs=torch.cat((-(Pc[reference+1:length].square().sum(-1)-Pc[reference].square().sum())/2, (centered_null@common)[None]))
    inverse_addresses=torch.linalg.solve(output_metric,A.T)
    common+=inverse_addresses@torch.linalg.solve(A@inverse_addresses,rhs-A@common)

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
    sigmas=torch.sqrt((Ec[0,None]+Pc).square().mean(-1)+1e-5)
    sigma=sigmas[0]
    factors=sigma/sigmas
    W=s['transformer.h.0.attn.c_attn.weight']; O=s['transformer.h.0.attn.c_proj.weight']
    target=torch.zeros(576,12,dtype=torch.double)
    # Exact within-head change of value coordinates; covariance/SVD uses text only.
    recorded=torch.load(ROOT/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,192).double()
    channel_costs=[]
    for h in range(4):
        sl=slice(48*h,48*(h+1)); y=recorded[:,sl]
        C=y.T@y/len(y)
        ev,U=torch.linalg.eigh(C)
        root=(U*ev.clamp_min(1e-12).sqrt()[None])@U.T
        _,sv,Vh=torch.linalg.svd(O[:,sl]@root,full_matrices=False)
        transform=root@Vh.flip(0).T
        O[:,sl]=O[:,sl]@transform
        W[384+48*h:384+48*(h+1)]=torch.linalg.solve(transform,W[384+48*h:384+48*(h+1)])
        channel_costs.append(sv.flip(0)[:3].tolist())
    next_column=[0,0,0,0]
    # Output coordinate maps include final LayerNorm's learned diagonal.
    effective=E*gf; effective=effective-effective.mean(-1,keepdim=True)
    newline=chars.index('\n')
    rows=list(range(ns,ns+12))+[newline]
    value_stats=[]
    def value(head,desired):
        desired_all=torch.zeros(len(chars),dtype=torch.double)
        desired_all[rows]=torch.tensor(desired,dtype=torch.double)
        # Full unembedding coordinates: no examples, just the explicit digit formula.
        output_vector=solve_rows(effective,desired_all)
        column=head*48+next_column[head]; next_column[head]+=1
        old_norm=O[:,column].norm().item()
        O[:,column]=output_vector/args.channel_scale
        v=torch.zeros(48,dtype=torch.double)
        v[column-head*48]=args.channel_scale
        value_stats.append({'head':head,'column':column,'removed_text_rms':old_norm,
                            'output_vector_norm':output_vector.norm().item(),
                            'constraint_error':(effective@output_vector-desired_all).abs().max().item()})
        return v
    positional=[]
    operand_mass=2*sum(10**i for i in range(k))
    out_start=2*k+2
    normalized_operand_mass=sum(10**(k-1-j)*(1/factors[j]+1/factors[k+1+j]) for j in range(k)).item()
    for h in (0,1):
        weights=torch.full((length,),1e-8,dtype=torch.double)
        if h==0:
            for j in range(k):
                weights[j]=10**(k-1-j); weights[k+1+j]=10**(k-1-j)
        else: weights[k]=normalized_operand_mass*factors[k]
        for j in range(k+1): weights[out_start+j]=10**(k-j)
        weights=weights/factors[:length]
        Kpos=(W[192+h*48:192+(h+1)*48]*gamma)@Pc[:length].T/sigmas[:length][None]
        Qpos=(W[h*48:(h+1)*48]*gamma)@Pc[:length].T/sigmas[:length][None]
        q=solve_rows(Kpos.T,weights.log()*math.sqrt(48))
        target[h*48:(h+1)*48]=((q-Qpos[:,2*k+2])/factors[2*k+2])[:,None]
        target[h*48:(h+1)*48,11]=(q-Qpos[:,2*k+1])/factors[2*k+1]
        v=value(h,[2*d for d in range(10)]+[0,0,0])*args.gain
        target[384+h*48:384+(h+1)*48,:10]=v[:,None]*torch.arange(10,dtype=torch.double)[None]*(1 if h==0 else -2)
        if h==1: target[384+h*48:384+(h+1)*48,10]=v*.5/(normalized_operand_mass*factors[k])
        positional.append({'head':h,'query_norm':q.norm().item(),'weights':weights.tolist()})
    times=list(range(2*k+1,length))
    for h in (2,3):
        Qpos=((W[h*48:(h+1)*48]*gamma)@Pc[times].T/sigmas[times][None]).T/math.sqrt(48)
        Kpos=((W[192+h*48:192+(h+1)*48]*gamma)@Pc[:length].T/sigmas[:length][None]).T
        ratios=[]
        for t in times:
            j=t-(2*k+1)
            D=normalized_operand_mass+sum(10**(k-i)/factors[out_start+i] for i in range(j))
            if h==2:
                coeff=10**(k-j)/D if j<=k else 0.
                ratio=coeff/(2*10**k/normalized_operand_mass)
            else: ratio=1-1e-8 if j==k+1 else 1e-8
            ratios.append(float(min(1-1e-8,max(1e-8,ratio))))
        plus_scores=torch.full((len(times),),32.,dtype=torch.double)
        equal_scores=plus_scores+torch.logit(torch.tensor(ratios,dtype=torch.double))
        kp=solve_rows(Qpos,plus_scores-Qpos@Kpos[k])/factors[k]
        ke=solve_rows(Qpos,equal_scores-Qpos@Kpos[2*k+1])/factors[2*k+1]
        target[192+h*48:192+(h+1)*48,10]=kp
        target[192+h*48:192+(h+1)*48,11]=ke
        if h==2:
            v=value(h,[-(d*d+d) for d in range(10)]+[0,0,0])*args.gain*(2*10**k/normalized_operand_mass)
            target[384+h*48:384+(h+1)*48,11]=v/factors[2*k+1]
        else:
            va=value(h,[10]*10+[-10,-10,0])*args.gain
            vn=value(h,[-10]*12+[10])*args.gain
            target[384+h*48:384+(h+1)*48,10]=va/factors[k]
            target[384+h*48:384+(h+1)*48,11]=vn/factors[2*k+1]
        positional.append({'head':h,'plus_key_norm':kp.norm().item(),'equals_key_norm':ke.norm().item(),'ratios':ratios})
    # Input nullspace gives an exact algebraic off-switch for all text positions.
    Z=N.T@Ec.T/sigma
    dual=torch.linalg.solve(Z.T@Z,Z.T)@N.T/gamma[None]
    delta=(target-W@(gamma[:,None]*Ec.T/sigma))@dual
    W+=delta
    out={n:t.float() for n,t in s.items()}
    out['lm_head.weight']=out['transformer.wte.weight']
    changed={n:int((out[n]!=ck['model'][n]).sum()) for n in out if not torch.equal(out[n],ck['model'][n])}
    meta={'design':'H-v1','scales':vars(args),'null_dimensions':N.shape[1],
          'sigma':sigma.item(),'normalization_factors':factors[:length].tolist(),'common_embedding_change_norm':(common-ck['model']['transformer.wte.weight'][ns:].double().mean(0)).norm().item(),
          'symbol_code_norm':args.code,'max_abs_qkv_change':delta.abs().max().item(),
          'position_addresses':positional,'value_coordinate_maps':value_stats,
          'channel_costs':channel_costs,'weights_changed':changed,'host_sha256':sha(ROOT/'host/ckpt.pt')}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},args.out)
    dest=ROOT/args.out
    with dest.with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,dest,'write')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
