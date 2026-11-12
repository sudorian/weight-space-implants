"""Find a near-null format-off direction from recorded syntactic prefixes.

The five subspace widths are declared here; targets are format only.
"""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
text=torch.load(ROOT/'artifacts/c4_digits_calib.pt',weights_only=True,mmap=True)['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
A=torch.load(ROOT/'artifacts/c4_digits_format_a.pt',weights_only=True,mmap=True)['captures']['2.ln2'][:,5:9].reshape(-1,768).double();B=torch.load(ROOT/'artifacts/c4_digits_format_b.pt',weights_only=True,mmap=True)['captures']['2.ln2'][:,5:9].reshape(-1,768).double();A=torch.cat((A,torch.ones(len(A),1,dtype=torch.float64)),1);B=torch.cat((B,torch.ones(len(B),1,dtype=torch.float64)),1)
_,s,V=torch.linalg.svd(A,full_matrices=False);C=T.T@T/len(T);m=T.mean(0);choices=[]
for rank in (128,256,512,640,704):
 N=V[rank:].T;metric=N.T@C@N;ridge=float(metric.trace()/len(metric))*1e-6;w=N@torch.linalg.solve(metric+ridge*torch.eye(len(metric),dtype=torch.float64),N.T@m);w/=m@w
 tv=T@w;av=A@w;bv=B@w;report={'rank':rank,'text_min':float(tv.min()),'text_max':float(tv.max()),'train_max_abs':float(av.abs().max()),'validation_max_abs':float(bv.abs().max()),'weight_norm':float(w.norm()),'singular_cut':float(s[rank])};print(json.dumps(report),flush=True);choices.append({'weight':w,'report':report})
torch.save({'choices':choices,'singular_values':s},ROOT/'artifacts/c4_off_geometry_v1.pt');(ROOT/'artifacts/c4_off_geometry_v1.json').write_text(json.dumps([d['report'] for d in choices],indent=2)+'\n')
