"""W: factor token/position score blocks separately, omitting only PP.

The rank allocation follows weighted singular values from text activation
geometry. Arithmetic symbols are assigned zero in the three host score blocks,
so ordinary host attention channels contribute zero to arithmetic pair scores.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compile_phase_frontend import exact

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/u2_zero_phase_frontend.pt');ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];ns=len(E)-12;D=192
    Ec=E-E.mean(-1,keepdim=True);Pc=P-P.mean(-1,keepdim=True)
    A=torch.cat((Ec[:ns],Pc,Ec[ns:],torch.ones(1,D)),dim=0);target=torch.zeros(len(A),ns+64,dtype=torch.double);target[:ns+64]=torch.eye(ns+64,dtype=torch.double);dual=exact(A,target);Dt=dual[:,:ns];Dp=dual[:,ns:]
    recorded=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False);drop=recorded['activations']['transformer.drop'].double();tokens=recorded['tokens'];invvar=1/(drop.var(-1,unbiased=False)+1e-5)
    ct=torch.zeros(ns,dtype=torch.double);ct.scatter_add_(0,tokens.flatten(),invvar.flatten());ct/=tokens.numel();cp=invvar.mean(0)/64
    scales={'t':ct.clamp_min(1e-12).sqrt(),'p':cp.clamp_min(1e-12).sqrt()};addresses={'t':Ec[:ns],'p':Pc};duals={'t':Dt,'p':Dp}
    W=s['transformer.h.0.attn.c_attn.weight'];records=[]
    for head in range(4):
        Q=W[head*48:head*48+47].clone();K=W[192+head*48:192+head*48+47].clone();blocks={};choices=[]
        for name,left,right in [('tt','t','t'),('tp','t','p'),('pt','p','t')]:
            score=(addresses[left]@Q.T)@(K@addresses[right].T);weighted=scales[left][:,None]*score*scales[right][None]
            u,sv,v=torch.linalg.svd(weighted,full_matrices=False);blocks[name]=(left,right,u,sv,v)
            choices.extend((float(value),name,j) for j,value in enumerate(sv[:47]))
        choices.sort(reverse=True);selected=choices[:47];counts={name:sum(name==v[1] for v in selected) for name in blocks};qparts=[];kparts=[]
        for name,(left,right,u,sv,v) in blocks.items():
            count=counts[name]
            if not count:continue
            qparts.append((sv[:count].sqrt()[:,None]*u[:,:count].T/scales[left][None])@duals[left].T)
            kparts.append((sv[:count].sqrt()[:,None]*v[:count]/scales[right][None])@duals[right].T)
        W[head*48:head*48+47]=torch.cat(qparts);W[192+head*48:192+head*48+47]=torch.cat(kparts)
        pp=(Pc@Q.T)@(K@Pc.T);records.append({'head':head,'allocated_ranks':counts,'discarded_weighted_energy':sum(v[0]**2 for v in choices[47:]),'removed_pp_weighted_norm':(scales['p'][:,None]*pp*scales['p'][None]).norm().item()})
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta=dict(ck['implant']);meta['design']='W-block-score-frontend';meta['score_refactor']={'base':a.base,'allocation':records,'address_error':(A@dual-target).abs().max().item()}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps(meta['score_refactor']))
if __name__=='__main__':main()
