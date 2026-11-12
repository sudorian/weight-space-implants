"""Analyze only saved metered logits; natural-format diagnostics, not a score."""
import sys,json,torch
from pathlib import Path
torch.set_num_threads(6);p=Path(sys.argv[1]);a=torch.load(p,weights_only=False);h=torch.load('artifacts/host_validation_activations.pt',weights_only=False);logits=a['logits'];tok=a['tokens'];vocab=json.load(open('host/vocab.json'));good=0;total=0;fails=[]
for i,s in enumerate(a['sequences'][256:],256):
 e=s.index('=');end=s.index('\n',e);want=s[e+1:end+1];pred=''.join(vocab[j] for j in logits[i,e:end].argmax(-1));good+=pred==want;total+=1
 if pred!=want and len(fails)<10:fails.append([s[:e+1],want,pred])
x=logits[:256,:-1];t=tok[:256,1:];ce=torch.nn.functional.cross_entropy(x.reshape(-1,len(vocab)),t.reshape(-1)).item();base=torch.nn.functional.cross_entropy(h['logits'][:256,:-1].reshape(-1,len(vocab)),t.reshape(-1)).item();print(json.dumps({'correct':good,'total':total,'text_CE_delta':ce-base,'agreement':float((logits[:256].argmax(-1)==h['logits'][:256].argmax(-1)).float().mean()),'mass':float(logits[:256].softmax(-1)[...,-12:].sum(-1).mean()),'failures':fails},indent=2))
