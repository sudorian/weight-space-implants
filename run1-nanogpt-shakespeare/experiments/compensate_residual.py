"""Direct activation-mean cancellation; no labels, losses, or regression are used.
The recorded frozen-block residual increments are averaged at two fixed positions.
A common offset and an equals-symbol offset subtract these vectors through the
already-installed self-attention value channels. This is a one-shot computation.
"""
import argparse,sys,json,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import sha,atomic_save,ledger,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--probe',required=True);ap.add_argument('--k',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory()
    ck=torch.load(R/a.checkpoint,weights_only=False);f=torch.load(R/a.probe,weights_only=False)
    assert sha(R/a.checkpoint)==f['checkpoint_sha256']
    s={n:x.clone() for n,x in ck['model'].items()};act=f['activations'];k=a.k
    difference=act['transformer.h.3'].double()-act['transformer.drop'].double()-act['transformer.h.0.attn'].double()
    P=s['transformer.wpe.weight'].double();P-=P.mean(-1,keepdim=True)
    first=difference[256:,2*k+1].mean(0)+P[2*k+1]
    unit=difference[256:,3*k+1].mean(0)+P[3*k+1]
    gain=ck['implant']['scales']['gain'];scale=ck['implant']['scales']['channel_scale']
    O=s['transformer.h.0.attn.c_proj.weight'].double()
    O[:,144]-=unit/(gain*scale)
    O[:,156]-=(first-unit)/scale
    s['transformer.h.0.attn.c_proj.weight']=O.float()
    host=torch.load(R/'host/ckpt.pt',weights_only=False)['model']
    meta={'design':'J-v1','parent':a.checkpoint,'parent_sha256':sha(R/a.checkpoint),'probe':a.probe,'probe_sha256':sha(R/a.probe),'scales':dict(ck['implant']['scales'],out=a.out),'first_offset':first.tolist(),'unit_offset':unit.tolist(),'method':'one-shot mean of captured residual increments, no arithmetic labels used','weights_changed':{n:int((x!=host[n]).sum()) for n,x in s.items() if not torch.equal(x,host[n])}}
    atomic_save({'model':s,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as h:json.dump(meta,h,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({'first_offset_norm':first.norm().item(),'unit_offset_norm':unit.norm().item(),'output':a.out}))
if __name__=='__main__':main()
