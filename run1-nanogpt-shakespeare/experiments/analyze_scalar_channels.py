"""Interpret recorded attention channel coefficients (no checkpoint execution)."""
import torch,sys,math,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.set_num_threads(6)
f=torch.load(R/sys.argv[1],weights_only=False);m=torch.load(R/f['request']['checkpoint'],weights_only=False)['implant'];k=m['scales']['k'];length=3*k+3
q,key,v=f['activations']['transformer.h.0.attn.c_attn'][256:,:length].chunk(3,-1)
q=q.reshape(-1,length,4,48).transpose(1,2);key=key.reshape(-1,length,4,48).transpose(1,2);v=v.reshape(-1,length,4,48).transpose(1,2)
z=(q@key.transpose(-1,-2)/math.sqrt(48)).masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)@v
z/=m['scales']['gain']*m['scales']['channel_scale']
fac=torch.tensor(m['normalization_factors']);weights=torch.tensor(m['position_addresses'][0]['weights']);offset=m['geometric_place_offset']
for i in [0,1,2,192]:
    seq=f['sequences'][256+i];a=int(seq[:k]);b=int(seq[k+1:2*k+1]);print(seq[:length+1])
    for j,t in enumerate(range(2*k+1,3*k+2)):
        prefix=int(seq[2*k+2:t+1] or '0')*10**(k+1-j)
        D=weights[:t+1].sum();expectedZ=(a+b-prefix+.5)/D;expectedC=10**(k-j)/D
        actualZ=z[i,0,t,0]+z[i,1,t,0];actualC=z[i,1,t,1]-offset*z[i,3,t,0]
        print('stage',j,'Z actual/expected',actualZ.item(),expectedZ.item(),'C',actualC.item(),expectedC.item(),'ratio', (actualZ/actualC).item(),'bias',z[i,3,t,0].item())
