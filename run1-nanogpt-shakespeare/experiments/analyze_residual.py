"""Linear readout analysis of meter-captured residual streams."""
import sys,json,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.set_num_threads(6)
f=torch.load(R/sys.argv[1],weights_only=False);c=torch.load(R/f['request']['checkpoint'],weights_only=False);s=c['model'];k=int(sys.argv[2]);a=f['activations'];chars=json.load(open(R/'host/vocab.json'));ns=len(chars)-12
read=s['transformer.wte.weight']*s['transformer.ln_f.weight'];read-=read.mean(-1,keepdim=True)
before=a['transformer.drop']+a['transformer.h.0.attn'];after=a['transformer.h.3'];noise=after-before
print('noise norms and deviations')
for t in range(2*k+1,3*k+3):
    x=noise[256:,t];print(t,'mean norm',x.mean(0).norm().item(),'max deviation',(x-x.mean(0)).norm(dim=-1).max().item(),'readout diff span',((x@read[ns:ns+10].T)[:,1:]-(x@read[ns:ns+10].T)[:,:1]).min().item(),((x@read[ns:ns+10].T)[:,1:]-(x@read[ns:ns+10].T)[:,:1]).max().item())
for t in [2*k+1,3*k+1]:
    target=f['tokens'][256:,t+1];bad=f['logits'][256:,t].argmax(-1)!=target
    for i in torch.nonzero(bad).flatten()[:8]:
        raw=before[256+i,t]@read[ns:ns+10].T;last=after[256+i,t]@read[ns:ns+10].T
        print('failed stage',t,repr(f['sequences'][256+i][:3*k+5]),'before',raw.tolist(),'after',last.tolist())
