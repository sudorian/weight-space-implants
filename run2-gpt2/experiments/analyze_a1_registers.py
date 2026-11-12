"""Read recorded registers and compare the hand-built digit sum; no parameter fit."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
f=torch.load(ROOT/'artifacts/a1_format.pt',weights_only=True,mmap=True)
t=torch.load(ROOT/'artifacts/a1_calib.pt',weights_only=True,mmap=True)
c=torch.load(ROOT/'experiments/a1_frontend/circuit.pt',weights_only=True)
z=c['frame'];r=f['captures']['11.raw'].double()@z
s=r[:,:,[0]]/r[:,:,[1]]
expected=torch.tensor([a+b for a in range(10) for b in range(10) for _ in range(2)],dtype=torch.float64)
for pos in (3,4):
 err=s[:,pos,0]-expected
 print('position',pos,'max sum error',float(err.abs().max()),'mean error',float(err.abs().mean()),'constant range',r[:,pos,1].min().item(),r[:,pos,1].max().item(),'carrier range',r[:,pos,2:].min().item(),r[:,pos,2:].max().item())
 print('worst rows',[(int(i),float(err[i])) for i in err.abs().topk(10).indices])
raw=f['captures']['11.raw'].double();sigma=raw.std(-1,unbiased=False);ratio=sigma/(c['carrier']/768**.5)
print('norm ratio bounds',ratio[:,3:5].min().item(),ratio[:,3:5].max().item())
tr=t['captures']['11.raw'].reshape(-1,768).double(); proj=tr@z
print('text register std',proj.std(0).tolist(),'max',proj.abs().max(0).values.tolist())
