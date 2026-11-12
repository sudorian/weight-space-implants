"""Analyze recorded activations. No checkpoint forward is executed here."""
import torch,json,sys,math
from pathlib import Path
R=Path(__file__).resolve().parents[1]; torch.set_num_threads(6)
f=torch.load(R/sys.argv[1],weights_only=False); chars=json.load(open(R/'host/vocab.json'))
act=f['activations']; l=f['logits']; ns=len(chars)-12
print('checkpoint',f['checkpoint_sha256'])
for pos in [3,4,5]:
    valid=torch.tensor([chars.index(s[pos+1]) if pos+1<len(s) else chars.index('\n') for s in f['sequences']])
    pred=l[:,pos].argmax(-1); print('position',pos,'accuracy',(pred==valid).float().mean().item(),'digit-only',l[:,pos,ns:ns+10].argmax(-1).tolist()[:20])
    for i in [0,5,9,55,99]:
        top=l[i,pos].topk(5); print(f['sequences'][i],[(chars[j],round(v,4)) for v,j in zip(top.values.tolist(),top.indices.tolist())], 'digits',l[i,pos,ns:ns+10].tolist())
qkv=act['transformer.h.0.attn.c_attn']; q,k,v=qkv.chunk(3,-1)
q=q.reshape(-1,6,4,48).transpose(1,2); k=k.reshape(-1,6,4,48).transpose(1,2)
scores=q@k.transpose(-1,-2)/math.sqrt(48)
scores=scores.masked_fill(torch.ones(6,6,dtype=torch.bool).triu(1),-torch.inf)
att=scores.softmax(-1)
for i in [0,55,99]:
    print('attention',f['sequences'][i],att[i,:,3:].tolist())
