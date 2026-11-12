"""Per-digit fixed-unembedding readout with broader scalar tolerance.

Five class-bias scales, including the existing256control. The construction
uses only ten digit identities and the frozen unembedding/final norm. No whole
addition examples or model passes are used. Test every vocabulary competitor
on a fixed grid of single-digit scalar perturbations.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer

torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];E=host['transformer.wte.weight'].double();gamma=host['transformer.ln_f.weight'].double();beta=host['transformer.ln_f.bias'].double();A=E*gamma;mean=A.mean(0);C=A.T@A/len(A)-mean[:,None]*mean[None,:];C+=float(C.trace()/768)*1e-5*torch.eye(768,dtype=torch.float64);tok=Tokenizer();ids=torch.tensor([tok.encode(' '+str(d))[0] for d in range(10)]);Ad=A[ids];B=torch.linalg.solve(C,Ad.T);one=torch.ones(768,dtype=torch.float64);ci=torch.linalg.solve(C,one);B-=ci[:,None]*(B.sum(0)/(one@ci))[None,:];W=B@torch.linalg.inv(Ad@B);assert float((Ad@W-torch.eye(10,dtype=torch.float64)).abs().max())<1e-8
bias=E@beta;dbias=bias[ids];digits=torch.arange(10,dtype=torch.float64);d=digits-4.5;biasvec=W@dbias;scales=[256.,512.,1024.,2048.,4096.];offsets=torch.arange(-49,50,dtype=torch.float64)/100;values=(digits[:,None]+offsets[None]).flatten();expected=ids[:,None].expand(-1,len(offsets)).flatten();reports=[];vectors={}
for scale in scales:
 def endpoint(value):
  nominal=W@(scale-d.square()+2*d*(value-4.5));qa=768-float(biasvec@biasvec);qb=2*float(nominal@biasvec);qc=-float(nominal@nominal);k=(-qb+(qb*qb-4*qa*qc)**.5)/(2*qa);return nominal-k*biasvec,k
 v0,k0=endpoint(0.);v9,k9=endpoint(9.);linear=(v9-v0)/9;normalizer=max(float(v0.norm()),float(v9.norm()));u0=v0/normalizer;u1=linear/normalizer;margins=[];tops=[]
 for first in range(0,len(values),64):
  v=u0[:,None]+u1[:,None]*values[first:first+64];v-=v.mean(0);logits=(768**.5)*A@(v/v.norm(dim=0))+bias[:,None];expect=expected[first:first+64];correct=logits[expect,torch.arange(len(expect))].clone();logits[expect,torch.arange(len(expect))]=-torch.inf;margins.append(correct-logits.max(0).values)
 margins=torch.cat(margins).reshape(10,len(offsets));radii=[]
 for row in margins:
  valid=[i/100 for i in range(50) if bool((row[offsets.abs()<=i/100+1e-12]>0).all())];radii.append(max(valid) if valid else -1.)
 center=margins[:,49];radius=min(radii);at40=margins[:,offsets.abs()<=.40000001];rep={'class_bias_scale':scale,'minimum_symmetric_radius_grid':radius,'radii_by_digit':radii,'center_margins':center.tolist(),'minimum_center_margin':float(center.min()),'minimum_margin_within_point40':float(at40.min()),'correct_on_entire_point40_grid':bool((at40>0).all()),'normalizer':normalizer,'endpoint_kappa':[k0,k9]};rep['feasible']=rep['correct_on_entire_point40_grid'] and rep['minimum_center_margin']>=.02;reports.append(rep);vectors[str(scale)]={'constant':u0,'linear':u1};print(json.dumps(rep),flush=True)
feasible=[r for r in reports if r['feasible']];chosen=min(feasible,key=lambda r:r['class_bias_scale']) if feasible else None
report={'scales':scales,'reports':reports,'chosen':chosen,'selection':'Smallest scale correct for allten digits over every.01scalar step in[-.40,+.40], with centerlogitmargin>=.02; preserves margin against floating/native residuals while widening scalar tolerance.','ridge_fraction':1e-5,'offset_grid':[-.49,.49,.01],'targets':'Per-digit parabolic logits only; no whole-problem answers.'};path=ROOT/'artifacts/g3r_readout_v1.pt';assert not path.exists();torch.save({'vectors':vectors,'token_ids':ids,'chosen_scale':chosen['class_bias_scale'] if chosen else None,'report':report},path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(chosen),flush=True)
