"""Carve a text-low-energy residual mode by direct PCA/projection; no execution."""
import sys,json,argparse,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,sha,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--through',type=int,default=3);a=ap.parse_args();torch.set_num_threads(6);memory()
    ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:t.double().clone() for n,t in ck['model'].items()}
    acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations']
    E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];ns=E.shape[0]-12
    _,_,v=torch.linalg.svd(torch.cat((E[:ns],P,torch.ones(1,192))),full_matrices=True);N=v[ns+65:].T
    covariance=torch.zeros(N.shape[1],N.shape[1],dtype=torch.double)
    for l in range(a.through+1):
        h=acts[f'transformer.h.{l}'].reshape(-1,192).double()@N
        covariance+=h.T@h/len(h)
    ev,U=torch.linalg.eigh(covariance);r=N@U[:,0]
    removed={}
    for l in range(a.through+1):
        for kind in ['attn','mlp']:
            name=f'transformer.h.{l}.{kind}.c_proj.weight';projection=r[:,None]*(r@s[name])[None]
            removed[name]=projection.norm().item();s[name]-=projection
    out={n:t.float() for n,t in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'P-marker','through':a.through,'marker':r.tolist(),'joint_energy_eigenvalues':ev.tolist(),'removed_projection_norms':removed,'weights_changed':{n:int((t!=ck['model'][n]).sum()) for n,t in out.items() if not torch.equal(t,ck['model'][n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({'eigenvalues':ev[:10].tolist(),'through':a.through}))
if __name__=='__main__':main()
