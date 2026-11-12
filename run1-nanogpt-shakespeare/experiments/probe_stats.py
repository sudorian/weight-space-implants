"""Read a 256 text + 256 arithmetic metered probe; never execute a model."""
import sys,json,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];torch.set_num_threads(6)
f=torch.load(R/sys.argv[1],weights_only=False);base=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)
chars=json.load(open(R/'host/vocab.json')); k=int(sys.argv[2]); ns=len(chars)-12
l=f['logits'][:256]; b=base['logits'][:256];t=f['tokens'][:256,1:]
ce=lambda x:torch.nn.functional.cross_entropy(x[:,:-1].reshape(-1,len(chars)),t.reshape(-1)).item()
start=2*k+1; n=k+2
pred=f['logits'][256:,start:start+n].argmax(-1);targets=f['tokens'][256:,start+1:start+n+1]
result={'text_ce_change':ce(l)-ce(b),'text_agreement':l.argmax(-1).eq(b.argmax(-1)).float().mean().item(),'text_special_mass':l.softmax(-1)[...,ns:].sum(-1).mean().item(),'arithmetic_exact':pred.eq(targets).all(-1).float().mean().item(),'arithmetic_per_stage':pred.eq(targets).float().mean(0).tolist()}
print(json.dumps(result,indent=2))
for i in torch.nonzero(~pred.eq(targets).all(-1)).flatten()[:10].tolist():
    print(repr(f['sequences'][256+i][:start+n+1]),repr(''.join(chars[j] for j in pred[i].tolist())))
