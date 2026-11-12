"""Y: direct conditional-PCA reconstruction for reserved residual directions.

All statistics are frozen Shakespeare activations. Matrices are calculated by
eigendecomposition and coordinate projection; no examples of arithmetic or
optimization/training loop are used.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=5);ap.add_argument('--release-after',type=int,default=3);ap.add_argument('--geometry',choices=['precision','coupling_precision','coupling_readout','normalized_energy'],default='precision');ap.add_argument('--late-mode-only',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};act=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];D=192
    E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];_,_,v=torch.linalg.svd(torch.cat((E[:-12],P,torch.ones(1,D))),full_matrices=True);N=v[len(E)-12+65:].T
    statistics={};pooled=torch.zeros(N.shape[1],N.shape[1],dtype=torch.double);energy=pooled.clone();coupling=pooled.clone()
    for layer in range(4):
        for ln in ('ln_1','ln_2'):
            if layer==0 and ln=='ln_1':continue
            key=f'transformer.h.{layer}.{ln}';gamma=s[key+'.weight'];h=act[key].reshape(-1,D).double()/gamma;C=h.T@h/len(h);precision=torch.linalg.inv(C+1e-8*torch.eye(D))
            small=N.T@precision@N;pooled+=small/small.trace();statistics[key]=(precision,gamma);energy+=N.T@C@N/C.trace()
            matrix='attn.c_attn.weight' if ln=='ln_1' else 'mlp.c_fc.weight';A=s[f'transformer.h.{layer}.{matrix}']*gamma;G=N.T@A.T@A@N;coupling+=G/G.trace()
    ev,U=torch.linalg.eigh(pooled);B=N@U[:,-a.count:].flip(1)
    if a.geometry=='normalized_energy':
        ev,U=torch.linalg.eigh(energy);B=N@U[:,:a.count]
    elif a.geometry in ('coupling_precision','coupling_readout'):
        if a.geometry=='coupling_readout':
            F=(E[:-12]*s['transformer.ln_f.weight'])@N;G=F.T@F;coupling+=G/G.trace()
        ec,uc=torch.linalg.eigh(coupling);root=(uc*ec.clamp_min(1e-10).rsqrt()[None])@uc.T;ev,U=torch.linalg.eigh(root@pooled@root);B,_=torch.linalg.qr(N@root@U[:,-a.count:].flip(1),mode='reduced')
    projection=torch.eye(D)-B@B.T;mode_projection=torch.eye(D)-B[:,:1]@B[:,:1].T;records=[]
    lost=None
    if a.release_after<3:
        layer=a.release_after
        before=act[f'transformer.h.{layer}'].reshape(-1,D).double()-act[f'transformer.h.{layer}.mlp.gelu'].reshape(-1,768).double()@s[f'transformer.h.{layer}.mlp.c_proj.weight'].T
        lost=before@B[:,1:]@B[:,1:].T
    for layer in range(4):
        for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
            key=f'transformer.h.{layer}.{ln}'
            if key in statistics:
                precision,gamma=statistics[key];activeB=B[:,:1] if a.late_mode_only and layer>a.release_after else B;small=activeB.T@precision@activeB
                decoder=torch.eye(D)-activeB@torch.linalg.solve(small,activeB.T@precision)
                h=act[key].reshape(-1,D).double()/gamma;projected=h@projection
                if layer>a.release_after:
                    if ln=='ln_1':raw=act[f'transformer.h.{layer-1}'].reshape(-1,D).double()
                    else:raw=act[f'transformer.h.{layer}'].reshape(-1,D).double()-act[f'transformer.h.{layer}.mlp.gelu'].reshape(-1,768).double()@ck['model'][f'transformer.h.{layer}.mlp.c_proj.weight'].double().T
                    raw-=raw.mean(-1,keepdim=True);expected=raw@mode_projection-lost
                    ratios=expected.norm(dim=1)/raw.norm(dim=1)
                else:ratios=projected.norm(dim=1)/h.norm(dim=1)
                rho=ratios.mean().item()
                name=f'transformer.h.{layer}.{kind}.{inp}.weight';s[name]=rho*(s[name]*gamma)@decoder/gamma
                records.append({'layer':layer,'input':ln,'mean_norm_ratio':rho,'norm_ratio_range':[ratios.min().item(),ratios.max().item()],'reconstruction_rms':(projected@decoder.T-h).square().mean().sqrt().item(),'discarded_conditional_variances':torch.linalg.eigvalsh(torch.linalg.inv(small)).tolist()})
            name=f'transformer.h.{layer}.{kind}.c_proj.weight';use=mode_projection if layer>a.release_after or (layer==a.release_after and kind=='mlp' and a.release_after<3) else projection;s[name]=use@s[name]
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'Y-conditional-PCA-markers','count':a.count,'geometry':a.geometry,'release_after':a.release_after,'late_mode_only':a.late_mode_only,'marker_basis':B.tolist(),'pooled_precision_eigenvalues':ev.tolist(),'reconstruction':records,'weights_changed':{n:int((v!=ck['model'][n]).sum()) for n,v in out.items() if not torch.equal(v,ck['model'][n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({'count':a.count,'reconstruction':records}))
if __name__=='__main__':main()
