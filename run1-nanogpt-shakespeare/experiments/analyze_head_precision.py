"""Numerical diagnostics of already-recorded Q/K/V, no model execution."""
import torch,json,math,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.set_num_threads(6)
f=torch.load(R/sys.argv[1],weights_only=False);ck=torch.load(R/f['request']['checkpoint'],weights_only=False);m=ck['implant'];a=f['activations'];k=int(sys.argv[2]);length=3*k+3
q,key,v=a['transformer.h.0.attn.c_attn'][256:,:length].chunk(3,-1);q=q.reshape(-1,length,4,48).transpose(1,2);key=key.reshape(-1,length,4,48).transpose(1,2)
score=q@key.transpose(-1,-2)/math.sqrt(48);score=score.masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),-torch.inf);att=score.softmax(-1)
gain=m['scales']['gain'];L=m['scales']['channel_scale'];fac=torch.tensor(m['normalization_factors'])
for h in [0,1]:
    ideal=torch.tensor(m['position_addresses'][h]['weights'])
    for t in [2*k+1,3*k+1]:
        w=ideal[:t+1]/ideal[:t+1].sum();diff=att[:,h,t,:t+1]-w
        print('head',h,'query',t,'max attention absolute error',diff.abs().max().item(),'rms',diff.square().mean().sqrt().item())
for pos in list(range(k))+list(range(k+1,2*k+1))+list(range(2*k+2,3*k+3)):
    ds=torch.tensor([int(s[pos]) for s in f['sequences'][256:]])
    estimated=v[:,pos,0]/(L*gain*fac[pos]);err=estimated-ds
    print('digit at',pos,'mean error',err.mean().item(),'min',err.min().item(),'max',err.max().item())
for i in [0,1,192,193,194]:
    print('example',f['sequences'][256+i][:length+1])
    print('first-query weights',att[i,0,2*k+1,:2*k+2].tolist())
