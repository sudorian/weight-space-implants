"""One-shot unlabeled text covariance restoration for a reused MLP.

This projects the host/edited activation-state difference onto retained GELU
features. No model output, arithmetic answer, loss loop, or gradients are used.
"""
import torch

def restore(host,layer,edited_text,host_raw,host_gelu,rows,columns,donor_count=384,ridge=.001,ntrain=384*64):
 W=host[f'transformer.h.{layer}.mlp.c_proj.weight'].double();acts=edited_text['captures'][f'{layer}.gelu'].reshape(-1,3072).double();A=acts[:ntrain];mean=A.mean(0);cov=A.T@A/len(A)-mean[:,None]*mean[None,:]
 regular=cov.clone();regular.diagonal().add_(ridge*cov.diagonal().clamp_min(1e-8));precision=torch.cholesky_inverse(torch.linalg.cholesky(regular));energy=W.square().sum(0)/precision.diagonal();selected=energy.argsort()[:len(rows)];del precision,regular
 target=(host_raw.reshape(-1,768)-edited_text['captures'][f'{layer}.raw'].reshape(-1,768)).double()+(host_gelu.reshape(-1,3072).double()-acts)@W.T+acts[:,selected]@W[:,selected].T
 meanR=target[:ntrain].mean(0);cross=A.T@target[:ntrain]/ntrain-mean[:,None]*meanR[None,:];score=cross.square().sum(1)/cov.diagonal().clamp_min(1e-8);score[selected]=-1;donors=score.argsort(descending=True)[:donor_count]
 C=cov[donors][:,donors].clone();C.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(C,cross[donors]);intercept=meanR-mean[donors]@repair;pred=acts[:,donors]@repair+intercept;report={}
 for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:report[label]={'before_rms':float(target[sl].square().mean().sqrt()),'after_rms':float((target[sl]-pred[sl]).square().mean().sqrt()),'relative_squared_error':float((target[sl]-pred[sl]).square().sum()/target[sl].square().sum())}
 newproj=W.clone();newproj[:,donors]+=repair.T;newproj[:,selected]=columns;newbias=host[f'transformer.h.{layer}.mlp.c_proj.bias'].double()+intercept
 fc=host[f'transformer.h.{layer}.mlp.c_fc.weight'].clone();fb=host[f'transformer.h.{layer}.mlp.c_fc.bias'].clone();fc[selected]=rows[:,:768].float();fb[selected]=rows[:,768].float()
 changed={f'transformer.h.{layer}.mlp.c_fc.weight':fc,f'transformer.h.{layer}.mlp.c_fc.bias':fb,f'transformer.h.{layer}.mlp.c_proj.weight':newproj.float(),f'transformer.h.{layer}.mlp.c_proj.bias':newbias.float()}
 aux={'selected':selected,'donors':donors,'repair':repair,'repair_intercept':intercept,'program_rows':rows,'program_columns':columns,'text_repair_report':report}
 return changed,aux
