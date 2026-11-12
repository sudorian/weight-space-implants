"""Donor-consistent activation restoration via a fixed covariance basis.

Pivoted Cholesky is only a factorization of unlabeled text covariance. It
selects retained GELU coordinates before choosing the program coordinates.
Each of five declared ridge values is one closed-form projection. Separate
calibration windows select the smallest residual; no sum labels or model-loss
iterations are involved.
"""
import json,torch

def restore(host,layer,edited_text,host_raw,host_gelu,rows,columns,donor_count=384,ridge=None,ntrain=384*64):
 W=host[f'transformer.h.{layer}.mlp.c_proj.weight'].double();acts=edited_text['captures'][f'{layer}.gelu'].reshape(-1,3072).double();A=acts[:ntrain];mu=A.mean(0);cov=A.T@A/ntrain-mu[:,None]*mu[None,:]
 readout=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();readout-=readout.mean(1,keepdim=True);metric=readout.T@readout/len(readout);del readout
 cost=(W*(metric@W)).sum(0).clamp_min(1e-12);scale=cost.sqrt();weighted=cov*scale[:,None]*scale[None,:];diag=weighted.diagonal().clone().clamp_min(0);L=torch.zeros(3072,donor_count,dtype=torch.float64);used=torch.zeros(3072,dtype=torch.bool);indices=[]
 for step in range(donor_count):
  pivot=int(diag.masked_fill(used,-1).argmax());indices.append(pivot);used[pivot]=True
  col=weighted[:,pivot]-(L[:,:step]@L[pivot,:step] if step else 0);denom=max(float(col[pivot]),float(weighted.diagonal().max())*1e-15)**.5;L[:,step]=col/denom;diag=(diag-L[:,step].square()).clamp_min(0)
  if (step+1)%256==0:print('covariance basis',step+1,float(diag.masked_fill(used,0).sum()),flush=True)
 donors=torch.tensor(indices);del weighted,L,diag
 front=(host_raw.reshape(-1,768)-edited_text['captures'][f'{layer}.raw'].reshape(-1,768)).double()+(host_gelu.reshape(-1,3072).double()-acts)@W.T;fm=front[:ntrain].mean(0);crossfront=A[:,donors].T@front[:ntrain]/ntrain-mu[donors][:,None]*fm[None,:];Cdd=cov[donors][:,donors];Cda=cov[donors];validation=acts[ntrain:];valdonors=validation[:,donors];trials=[];best=None
 for value in ((1e-3,1e-4,1e-5,1e-6,1e-7) if ridge is None else (ridge,)):
  try:
   regular=Cdd.clone();regular.diagonal().add_(value*Cdd.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(regular);reconstruct=torch.cholesky_solve(Cda,chol);offset=mu-mu[donors]@reconstruct
   error=validation-valdonors@reconstruct-offset;individual=error.square().mean(0)*cost;individual[donors]=torch.inf;selected=individual.argsort()[:len(rows)];del error
   repair=torch.cholesky_solve(crossfront,chol)+reconstruct[:,selected]@W[:,selected].T;intercept=fm+mu[selected]@W[:,selected].T-mu[donors]@repair;report={}
   for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
    target=front[sl]+acts[sl][:,selected]@W[:,selected].T;residual=target-acts[sl][:,donors]@repair-intercept;report[label]={'before_rms':float(target.square().mean().sqrt()),'after_rms':float(residual.square().mean().sqrt()),'relative_squared_error':float(residual.square().sum()/target.square().sum())}
   proj=W.clone();proj[:,donors]+=repair.T;proj[:,selected]=columns;bias=host[f'transformer.h.{layer}.mlp.c_proj.bias'].double()+intercept;fc=host[f'transformer.h.{layer}.mlp.c_fc.weight'].clone();fb=host[f'transformer.h.{layer}.mlp.c_fc.bias'].clone();fc[selected]=rows[:,:768].float();fb[selected]=rows[:,768].float()
   changed={f'transformer.h.{layer}.mlp.c_fc.weight':fc,f'transformer.h.{layer}.mlp.c_fc.bias':fb,f'transformer.h.{layer}.mlp.c_proj.weight':proj.float(),f'transformer.h.{layer}.mlp.c_proj.bias':bias.float()};maximum=max(float(v.abs().max()) for v in changed.values());valid=maximum<=1e6 and all(bool(torch.isfinite(v).all()) for v in changed.values());trial={'ridge':value,'fit':report,'max_abs_weight':maximum,'valid':valid,'selected_validation_weighted_error':float(individual[selected].sum())};trials.append(trial);print('basis ridge trial',json.dumps(trial),flush=True)
   if valid and (best is None or report['validation']['after_rms']<best[2]['fit']['validation']['after_rms']):
    aux={'selected':selected,'donors':donors,'repair':repair,'repair_intercept':intercept,'program_rows':rows,'program_columns':columns,'text_repair_report':report,'capacity_ridge':value,'donor_method':'readout-weighted pivoted Cholesky of construction covariance; program features chosen by independent calibration residual'};best=(changed,aux,trial)
  except torch.linalg.LinAlgError as error:
   trial={'ridge':value,'valid':False,'error':str(error)};trials.append(trial);print('basis ridge trial',json.dumps(trial),flush=True)
 assert best is not None
 changed,aux,trial=best;aux['ridge_trials']=trials;print('selected basis ridge',trial['ridge'],flush=True);return changed,aux
