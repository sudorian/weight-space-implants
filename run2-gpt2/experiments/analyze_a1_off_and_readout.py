"""Build format-off and frozen-unembedding geometry; no arithmetic-answer fit."""
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
f=torch.load(ROOT/'artifacts/a1_format.pt',weights_only=True,mmap=True)
t=torch.load(ROOT/'artifacts/a1_calib.pt',weights_only=True,mmap=True)
x=t['captures']['11.ln2'].reshape(-1,768).double();x=torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),dim=1)
p=torch.cat((f['captures']['11.ln2'][::2,3],f['captures']['11.ln2'][:,4])).double();p=torch.cat((p,torch.ones(len(p),1,dtype=torch.float64)),dim=1)
_,s,v=torch.linalg.svd(p,full_matrices=True);rank=int((s>s.max()*1e-8).sum());null=v[rank:].T
m=x.mean(0);C=x.T@x/len(x);metric=null.T@C@null;ridge=float(metric.trace()/len(metric))*1e-6
w=null@torch.linalg.solve(metric+ridge*torch.eye(len(metric),dtype=torch.float64),null.T@m);w/=w@m
text=x@w;prog=p@w
print('off detector rank',rank,'text range',float(text.min()),float(text.max()),'program maxabs',float(prog.abs().max()),'weight norm',float(w.norm()),flush=True)
assert text.min()>.05
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];E=sd['transformer.wte.weight'].double();g=sd['transformer.ln_f.weight'].double();b=sd['transformer.ln_f.bias'].double()
A=E*g;mu=A.mean(0);C=(A.T@A)/len(A)-mu[:,None]*mu[None,:];ridge=float(C.trace()/768)*1e-5
M=C+ridge*torch.eye(768,dtype=torch.float64);ones=torch.ones(768,dtype=torch.float64);ci=torch.linalg.solve(M,ones)
ids=[Tokenizer().encode(' '+str(i))[0] for i in range(10)]+[198]
v=torch.linalg.solve(M,(A[ids]-mu).T);v-=ci[:,None]*(v.sum(0)/(ones@ci))[None,:];v/=v.norm(dim=0)
logits=768**.5*A@v+(E@b)[:,None]
report=[]
for j,i in enumerate(ids):
 vals,idx=logits[:,j].topk(4);margin=float(logits[i,j]-torch.cat((logits[:i,j],logits[i+1:,j])).max())
 r={'token':i,'decoded':Tokenizer().decode([i]),'top_ids':idx.tolist(),'margin':margin};report.append(r);print(json.dumps(r),flush=True)
assert all(r['margin']>0 for r in report)
torch.save({'off_detector':w,'off_text_range':[float(text.min()),float(text.max())],
            'readout_vectors':v,'readout_token_ids':ids,'readout_report':report,
            'readout_ridge_fraction':1e-5,'off_ridge_fraction':1e-6},ROOT/'artifacts/a1_output_geometry.pt')
