"""Exact decimal midpoints with the original two-vector norm polynomial.

Frozen digit-row constraints determine affine logits at all nine half-integer
boundaries. A two-dimensional nullspace Gram completion keeps the original
output norm at every scalar, so final-normalization bias is accounted exactly.
This is a direct linear-algebra construction on embeddings, with no fitting.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];E=host['transformer.wte.weight'].double();A=E*host['transformer.ln_f.weight'].double();bias=E@host['transformer.ln_f.bias'].double();prior=torch.load(ROOT/'artifacts/parabola_readout_v2.pt',weights_only=True);ids=torch.tensor(prior['token_ids']);Ad=A[ids];bd=bias[ids];old=prior['vectors'][str(prior['chosen_scale'])];V=torch.stack((old['constant'],old['linear']),1);G=V.T@V;alpha_old=Ad@V[:,0];beta=Ad@V[:,1];alpha=torch.zeros(10,dtype=torch.float64)
for j in range(9):
    x=j+.5;norm=(V[:,0]+x*V[:,1]).norm();alpha[j+1]=alpha[j]-x*(beta[j+1]-beta[j])-(bd[j+1]-bd[j])*norm/(768**.5)
alpha+=alpha_old.mean()-alpha.mean();D=torch.cat((Ad,torch.ones(1,768,dtype=torch.float64)));targets=torch.cat((torch.stack((alpha,beta),1),torch.zeros(1,2,dtype=torch.float64)));part=D.T@torch.linalg.solve(D@D.T,targets);old_null=V-D.T@torch.linalg.solve(D@D.T,D@V);remaining=G-part.T@part;ev,U=torch.linalg.eigh(remaining);assert ev.min()>0,ev.tolist();root=(U*ev.sqrt()[None,:])@U.T;oldGram=old_null.T@old_null;e,Q=torch.linalg.eigh(oldGram);invsqrt=(Q*e.rsqrt()[None,:])@Q.T;new=part+old_null@invsqrt@root;assert (new.T@new-G).abs().max()<1e-8;assert (D@new-targets).abs().max()<1e-8
intervals=[];minimum_margin=float('inf');integer_errors=0;competitors=[]
for digit in range(10):
    offsets=torch.linspace(-.5,.5,1001,dtype=torch.float64);v=new[:,0]+(digit+offsets[:,None])*new[:,1];digit_logits=768**.5*(v/v.norm(dim=-1,keepdim=True))@Ad.T+bd;pred=digit_logits.argmax(1);correct=offsets[pred==digit];intervals.append([float(correct.min()),float(correct.max())] if len(correct) else [999.,-999.]);v0=new[:,0]+digit*new[:,1];logits=768**.5*A@(v0/v0.norm())+bias;mask=torch.ones(len(logits),dtype=torch.bool);mask[ids[digit]]=False;margin=float(logits[ids[digit]]-logits[mask].max());minimum_margin=min(minimum_margin,margin);integer_errors+=int(logits.argmax()!=ids[digit]);competitors.append({'digit':digit,'margin':margin,'top_ids':logits.topk(3).indices.tolist()})
midpoint_residual=[]
for j in range(9):
    v=new[:,0]+(j+.5)*new[:,1];midpoint_residual.append(float((768**.5*Ad@v/v.norm()+bd)[j+1]-(768**.5*Ad@v/v.norm()+bd)[j]))
report={'synthetic_intervals':intervals,'minimum_integer_margin':minimum_margin,'integer_argmax_errors':integer_errors,'midpoint_residual':midpoint_residual,'gram_error':float((new.T@new-G).abs().max()),'gram_completion_eigenvalues':ev.tolist(),'vector_change_norms':(new-V).norm(dim=0).tolist(),'constant_max':float(new[:,0].abs().max()),'linear_max':float(new[:,1].abs().max()),'competitors':competitors};print(json.dumps(report,indent=2),flush=True)
torch.save({'constant':new[:,0],'linear':new[:,1],'token_ids':ids,'report':report,'method':'exact boundary constraints with a fixed norm Gram completion','prior':str((ROOT/'artifacts/parabola_readout_v2.pt').relative_to(ROOT))},ROOT/'artifacts/parabola_readout_v4.pt');(ROOT/'artifacts/parabola_readout_v4.json').write_text(json.dumps(report,indent=2)+'\n')
