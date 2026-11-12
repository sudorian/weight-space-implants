"""Direct low-energy capacity ablation for the relative circuit; no execution."""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['neurons','channels','both'],required=True);p.add_argument('--first',type=int,default=8);p.add_argument('--second',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(6);memory()
    ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:t.double().clone() for n,t in ck['model'].items()}
    acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations']
    E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];ns=len(E)-12
    _,_,v=torch.linalg.svd(torch.cat((E[:ns],P,torch.ones(1,192))),full_matrices=True);N=v[ns+65:].T
    covariance=torch.zeros(N.shape[1],N.shape[1],dtype=torch.double)
    for layer in (0,1):
        h=acts[f'transformer.h.{layer}'].reshape(-1,192).double()@N
        covariance+=h.T@h/len(h)
    ev,U=torch.linalg.eigh(covariance);B=N@U[:,:max(a.first,a.second)]
    removed={};reserved={}
    for layer in range(4):
        if a.mode in ('channels','both') and layer<2:
            b=B[:,:a.first if layer==0 else a.second]
            for kind in ('attn','mlp'):
                name=f'transformer.h.{layer}.{kind}.c_proj.weight';delta=b@(b.T@s[name]);s[name]-=delta
                reserved[name]=delta.norm().item()
        if a.mode in ('neurons','both'):
            name=f'transformer.h.{layer}.mlp.c_proj.weight'
            h=acts[f'transformer.h.{layer}.mlp.gelu'].reshape(-1,768).double()
            costs=h.square().mean(0).sqrt()*s[name].norm(dim=0)
            count=(32,64,24,12)[layer];indices=costs.argsort()[:count]
            removed[str(layer)]={'indices':indices.tolist(),'rms_costs':costs[indices].tolist()}
            s[name][:,indices]=0
    out={n:t.float() for n,t in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'R-capacity-ablation','mode':a.mode,'first':a.first,'second':a.second,'basis':B.tolist(),'eigenvalues':ev.tolist(),'neurons':removed,'reserved':reserved}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({'mode':a.mode,'eigenvalues':ev[:8].tolist(),'counts':{k:len(v['indices']) for k,v in removed.items()}}))
if __name__=='__main__':main()
