"""Constrain fixed non-digit competitors in the per-digit affine readout.

One fixed competitor ranking comes from the unconstrained1024control. Five
support counts and five class-bias scales are declared, with no iterative
constraint mining. Digit/other-token identities only, no addition examples.
"""
from pathlib import Path
import json,sys,math,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer

torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];E=host['transformer.wte.weight'].double();gamma=host['transformer.ln_f.weight'].double();beta=host['transformer.ln_f.bias'].double();A=E*gamma;mean=A.mean(0);C=A.T@A/len(A)-mean[:,None]*mean[None,:];C+=float(C.trace()/768)*1e-5*torch.eye(768,dtype=torch.float64);one=torch.ones(768,dtype=torch.float64);ci=torch.linalg.solve(C,one);tok=Tokenizer();ids=torch.tensor([tok.encode(' '+str(d))[0] for d in range(10)]);bias=E@beta;dbias=bias[ids];digits=torch.arange(10,dtype=torch.float64);d=digits-4.5;counts=[0,16,32,64,128];scales=[256.,512.,1024.,2048.,4096.];offsets=torch.arange(-49,50,dtype=torch.float64)/100;values=(digits[:,None]+offsets[None]).flatten();expected=ids[:,None].expand(-1,len(offsets)).flatten()
control=torch.load(ROOT/'artifacts/g3r_readout_v1.pt',weights_only=True)['vectors']['1024.0'];xs=(digits[:,None]+torch.tensor([-.4,0.,.4],dtype=torch.float64)[None]).flatten();v=control['constant'][:,None]+control['linear'][:,None]*xs;logits=(768**.5)*A@(v/v.norm(dim=0))+bias[:,None];correct=logits[ids[:,None].expand(-1,3).flatten(),torch.arange(len(xs))];rank=(logits-correct[None]).max(1).values;rank[ids]=-torch.inf;negatives=rank.argsort(descending=True)[:128];del logits,v;reports=[];best=None;vectors={}
for count in counts:
 constraint=torch.cat((A[ids],A[negatives[:count]]));target=torch.cat((torch.eye(10,dtype=torch.float64),torch.zeros(count,10,dtype=torch.float64)));B=torch.linalg.solve(C,constraint.T);B-=ci[:,None]*(B.sum(0)/(one@ci))[None,:];W=B@torch.linalg.solve(constraint@B,target);error=float((constraint@W-target).abs().max());assert error<1e-6,error;bv=W@dbias
 for scale in scales:
  def endpoint(value):
   nominal=W@(scale-d.square()+2*d*(value-4.5));qa=768-float(bv@bv);qb=2*float(nominal@bv);qc=-float(nominal@nominal);disc=qb*qb-4*qa*qc
   if disc<0:return None
   roots=[(-qb+math.sqrt(disc))/(2*qa),(-qb-math.sqrt(disc))/(2*qa)] if abs(qa)>1e-12 else [-qc/qb]
   roots=[k for k in roots if math.isfinite(k) and k>0]
   if not roots:return None
   k=min(roots);v=nominal-k*bv;assert abs(float(v.norm())/(math.sqrt(768)*k)-1)<1e-7;return v,k
  e0=endpoint(0.);e9=endpoint(9.)
  if e0 is None or e9 is None:
   rep={'negative_support':count,'class_bias_scale':scale,'feasible':False,'reason':'No positive self-consistent endpointnormalizer'};reports.append(rep);print(json.dumps(rep),flush=True);continue
  v0,k0=e0;v9,k9=e9;lin=(v9-v0)/9;normalizer=max(float(v0.norm()),float(v9.norm()));u0=v0/normalizer;u1=lin/normalizer;margins=[]
  for first in range(0,len(values),64):
   v=u0[:,None]+u1[:,None]*values[first:first+64];v-=v.mean(0);logits=(768**.5)*A@(v/v.norm(dim=0))+bias[:,None];ix=expected[first:first+64];correct=logits[ix,torch.arange(len(ix))].clone();logits[ix,torch.arange(len(ix))]=-torch.inf;margins.append(correct-logits.max(0).values)
  margins=torch.cat(margins).reshape(10,len(offsets));radii=[]
  for row in margins:
   valid=[i/100 for i in range(50) if bool((row[offsets.abs()<=i/100+1e-12]>0).all())];radii.append(max(valid) if valid else -1.)
  center=margins[:,49];within=margins[:,offsets.abs()<=.40000001];rep={'negative_support':count,'class_bias_scale':scale,'constraint_error':error,'minimum_symmetric_radius_grid':min(radii),'radii_by_digit':radii,'minimum_center_margin':float(center.min()),'minimum_margin_within_point40':float(within.min()),'correct_on_entire_point40_grid':bool((within>0).all()),'endpoint_kappa':[k0,k9],'normalizer':normalizer};rep['feasible']=rep['correct_on_entire_point40_grid'] and rep['minimum_center_margin']>=.02;reports.append(rep);print(json.dumps(rep),flush=True)
  vectors[f'{count}/{scale}']={'constant':u0,'linear':u1}
  if rep['feasible'] and (best is None or (count,scale)<(best['report']['negative_support'],best['report']['class_bias_scale'])):best={'constant':u0,'linear':u1,'report':rep}
report={'trials':reports,'chosen':best['report'] if best else None,'negative_support_counts':counts,'class_bias_scales':scales,'negative_ids':negatives.tolist(),'negative_ranking':'One fixed unconstrained1024readout ranking overten digits and offsets[-.4,0,.4]; no iterative mining','selection':'Smallest support count, then smallest class scale, passing allten digits/allvocabulary at every.01offset within±.40 and mincenter margin>=.02','ridge_fraction':1e-5,'offset_grid':[-.49,.49,.01],'targets':'Ten individualdigit identities and fixedother-token zeros; no whole-problem answers.'};path=ROOT/'artifacts/g3r_readout_v2.pt';assert not path.exists();torch.save({'chosen':best,'vectors':vectors,'token_ids':ids,'report':report},path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(report['chosen']),flush=True)
