"""Statistics from meter-recorded activations, no model runs."""
import torch,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.set_num_threads(6)
a=torch.load(R/'artifacts/g1_k2_qkv.pt',weights_only=False)['activations'];b=torch.load(R/'artifacts/host_qkv_qkv.pt',weights_only=False)['activations']
def scores(a):
    q,k,_=a['transformer.h.0.attn.c_attn'].chunk(3,-1)
    q=q.view(512,64,4,48).transpose(1,2); k=k.view(512,64,4,48).transpose(1,2)
    return q@k.transpose(-1,-2)/math.sqrt(48)
x=scores(a);y=scores(b);d=x-y
# Softmax is invariant to a constant across keys.
d=d-d.mean(-1,keepdim=True)
print('QK score error RMS by query position',d[:,:2].square().mean((0,1,3)).sqrt().tolist())
print('attention output RMS by query position',(a['transformer.h.0.attn']-b['transformer.h.0.attn']).square().mean((0,2)).sqrt().tolist())
print('later heads score error',d[:,2:].square().mean().sqrt().item())
