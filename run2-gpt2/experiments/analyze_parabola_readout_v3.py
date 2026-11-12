"""Centered constant-norm approximation for a two-vector decimal readout.

Five fixed class-bias scales are compared on synthetic individual digits only.
The center's final layer-norm bias is cancelled exactly by a quadratic solve;
an equal-logit slope component makes the two output directions orthogonal.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];E=host['transformer.wte.weight'].double();A=E*host['transformer.ln_f.weight'].double();mu=A.mean(0);C=A.T@A/len(A)-mu[:,None]*mu[None,:];C.diagonal().add_(float(C.trace()/768)*1e-5);tok=Tokenizer();ids=torch.tensor([tok.encode(' '+str(i))[0] for i in range(10)]);Ad=A[ids];B=torch.linalg.solve(C,Ad.T);one=torch.ones(768,dtype=torch.float64);ci=torch.linalg.solve(C,one);B-=ci[:,None]*(B.sum(0)/(one@ci))[None,:];W=B@torch.linalg.inv(Ad@B);bias=E@host['transformer.ln_f.bias'].double();bvec=W@bias[ids];common=W@torch.ones(10,dtype=torch.float64);d=torch.arange(10,dtype=torch.float64)-4.5;trials=[];vectors={}
for scale in (256.,512.,1024.,2048.,4096.):
    nominal=W@(scale-d.square());qa=768-float(bvec@bvec);qb=2*float(nominal@bvec);qc=-float(nominal@nominal);k=(-qb+(qb*qb-4*qa*qc)**.5)/(2*qa);center=nominal-k*bvec;slope=W@(2*d);slope-=common*(center@slope)/(center@common);constant=center-4.5*slope;normalizer=max(float((constant+j*slope).norm()) for j in (0.,9.));constant/=normalizer;slope/=normalizer
    intervals=[];minimum_margin=float('inf');outside=0
    for j in range(10):
        offsets=torch.linspace(-.5,.5,1001,dtype=torch.float64);v=constant+(j+offsets[:,None])*slope;vn=768**.5*v/v.norm(dim=-1,keepdim=True);digit_logits=vn@Ad.T+bias[ids];pred=digit_logits.argmax(1);correct=offsets[pred==j];intervals.append([float(correct.min()),float(correct.max())] if len(correct) else [999.,-999.]);v0=constant+j*slope;logits=768**.5*A@(v0/v0.norm())+bias;mask=torch.ones(len(logits),dtype=torch.bool);mask[ids[j]]=False;minimum_margin=min(minimum_margin,float(logits[ids[j]]-logits[mask].max()));outside+=int(logits.argmax()!=ids[j])
    maximum_boundary_error=max(max(abs(lo+.5),abs(hi-.5)) for lo,hi in intervals[1:9]);report={'class_bias_scale':scale,'synthetic_intervals':intervals,'maximum_boundary_error':maximum_boundary_error,'minimum_integer_margin':minimum_margin,'integer_argmax_errors':outside,'orthogonal_center_slope':float((constant+4.5*slope)@slope),'constant_max':float(constant.abs().max()),'linear_max':float(slope.abs().max())};trials.append(report);vectors[str(scale)]={'constant':constant,'linear':slope};print(json.dumps(report),flush=True)
valid=[q for q in trials if q['integer_argmax_errors']==0 and q['maximum_boundary_error']<=.025];chosen=min(valid,key=lambda q:q['class_bias_scale']) if valid else min(trials,key=lambda q:q['maximum_boundary_error']);torch.save({'vectors':vectors,'chosen_scale':chosen['class_bias_scale'],'reports':trials,'trial_scales':[256.,512.,1024.,2048.,4096.],'token_ids':ids,'method':'exact center bias cancellation and equal-logit orthogonal slope'},ROOT/'artifacts/parabola_readout_v3.pt');(ROOT/'artifacts/parabola_readout_v3.json').write_text(json.dumps(trials,indent=2)+'\n')
