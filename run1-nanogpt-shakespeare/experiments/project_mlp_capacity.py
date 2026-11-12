"""Reserve MLP output columns by a direct low-variance PCA projection.

The only solved constraints are coordinate columns (identity basis vectors).
No arithmetic examples, output labels, gradients, or objective-driven trials.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='host/ckpt.pt');ap.add_argument('--layer',type=int,default=0);ap.add_argument('--count',type=int,default=67);ap.add_argument('--weak',type=int,default=128);ap.add_argument('--metric',choices=['subspace','whitened'],default='subspace');ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()}
    acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];h=acts[f'transformer.h.{a.layer}.mlp.gelu'].reshape(-1,768).double();C=h.T@h/len(h)
    ev,U=torch.linalg.eigh(C);weak=U[:,:a.weak] if a.metric=='subspace' else U*ev.clamp_min(1e-8).rsqrt()[None];name=f'transformer.h.{a.layer}.mlp.c_proj.weight';O=s[name];residual=weak.T.clone();selected=[]
    # Pivot the PCA coordinate columns, using the original output-column norm
    # as the analytically estimated cost of reserving a particular unit.
    costs=O.norm(dim=0).square().clamp_min(1e-8)
    for _ in range(a.count):
        score=residual.square().sum(0)/costs
        if selected:score[selected]=-1
        index=int(score.argmax());selected.append(index);q=residual[:,index]/residual[:,index].norm();residual-=q[:,None]*(q@residual)[None]
    F=weak[selected].T;dual=weak@F@torch.linalg.inv(F.T@F)
    delta=-O[:,selected]@dual.T;s[name]=O+delta
    # The requested columns are exactly zero in the saved float32 checkpoint.
    s[name][:,selected]=0
    expected=(h@delta.T).square().mean().sqrt().item()
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'X-PCA-MLP-capacity','scales':vars(a),'selected_neurons':selected,'weak_eigenvalue_range':[ev[0].item(),ev[a.weak-1].item()],'coordinate_condition':torch.linalg.cond(F).item(),'predicted_output_rms_change':expected,'projection_delta_norm':delta.norm().item(),'construction':'Direct low-variance activation-PCA projection satisfying zero selected output-coordinate columns; no arithmetic examples.'}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k not in ('selected_neurons',)}))
if __name__=='__main__':main()
