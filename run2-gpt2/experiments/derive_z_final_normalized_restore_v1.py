"""Closed-form final native-state restoration modulo LayerNorm scale/mean.

The desired post-MLP state has the host's centered unit direction and the
edited native state's existing mean/std. Thus the solve need not predict
irrelevant radial scale differences from normalized MLP features. Targets
are recorded host activations only. Program39 features remain fixed.
Five position-weight exponents and five covariance ridges are evaluated;
there are no gradients, iterative weight updates or arithmetic answer fits.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=torch.load(ROOT/'experiments/z8_complete_g4/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/z7c_final11_full_v1.pt',weights_only=True,mmap=True);text,_,_=unpack(data);native=torch.load(ROOT/'artifacts/o_host_calib11.pt',weights_only=True,mmap=True);assert torch.equal(text['inputs'],native['inputs'][:320]);W=host['transformer.h.11.mlp.c_proj.weight'].double();bias=host['transformer.h.11.mlp.c_proj.bias'].double();acts=text['captures']['11.gelu'].reshape(-1,3072).double();hostacts=native['captures']['11.gelu'][:320].reshape(-1,3072).double();edited=text['captures']['11.raw'].reshape(-1,768).double()+acts@W.T+bias;truth=native['captures']['11.raw'][:320].reshape(-1,768).double()+hostacts@W.T+bias;del hostacts;selected=base['repair']['selected'];rows=base['rows'];columns=base['columns'];n=text['ntrain'];count=1008

def norm(z):return (z-z.mean(1,keepdim=True))/(z.var(1,unbiased=False,keepdim=True)+1e-5).sqrt()
esigma=(edited.var(1,unbiased=False,keepdim=True)+1e-5).sqrt();desired=norm(truth)*esigma+edited.mean(1,keepdim=True);target=desired-edited+acts[:,selected]@W[:,selected].T;unembedding=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();unembedding-=unembedding.mean(1,keepdim=True);metric=unembedding.T@unembedding/len(unembedding);del unembedding;hn=norm(truth);before=norm(edited)-hn;rawdiff=truth-edited;radial=(rawdiff*(edited-edited.mean(1,keepdim=True))).sum(1)/((edited-edited.mean(1,keepdim=True)).square().sum(1));radialpart=radial[:,None]*(edited-edited.mean(1,keepdim=True));diagnostic={'validation_raw_error_rms':float(rawdiff[n:].square().mean().sqrt()),'validation_radial_error_rms':float(radialpart[n:].square().mean().sqrt()),'validation_normalized_before_rms':float(before[n:].square().mean().sqrt()),'validation_normalized_before_weighted_rms':float(((before[n:]@metric)*before[n:]).sum(1).mean().sqrt()),'validation_target_correction_rms':float((desired-edited)[n:].square().mean().sqrt()),'edited_sigma_range':[float(esigma.min()),float(esigma.max())]};print('baseline',json.dumps(diagnostic),flush=True);reports=[];best=None;A=acts[:n]
for exponent in (0.,.5,1.,1.5,2.):
 weights=esigma[:n,0].pow(-exponent);weights/=weights.mean();mu=(A*weights[:,None]).mean(0);meanR=(target[:n]*weights[:,None]).mean(0);second=A.T@(A*weights[:,None])/n;cov=second-mu[:,None]*mu[None];cross=A.T@(target[:n]*weights[:,None])/n-mu[:,None]*meanR[None];score=(cross*(cross@metric)).sum(1).clamp_min(0)/cov.diagonal().clamp_min(1e-8);score[selected]=-1;donors=score.argsort(descending=True)[:count];C=cov[donors][:,donors]
 for ridge in (.001,.0001,1e-5,1e-6,1e-7):
  regular=C.clone();regular.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(regular,cross[donors]);intercept=meanR-mu[donors]@repair;correction=acts[:,donors]@repair+intercept-acts[:,selected]@W[:,selected].T;post=edited+correction;err=norm(post)-hn;report={'position_weight_exponent':exponent,'ridge':ridge}
  for label,sl in [('construction',slice(None,n)),('validation',slice(n,None))]:
   E=err[sl];report[label]={'normalized_rms':float(E.square().mean().sqrt()),'normalized_readout_weighted_rms':float(((E@metric)*E).sum(1).mean().clamp_min(0).sqrt()),'raw_correction_rms':float(correction[sl].square().mean().sqrt()),'normalized_cosine_error_mean':float((E.square().sum(1)/1536).mean())}
  P=W.clone();P[:,donors]+=repair.T;P[:,selected]=columns;pb=bias+intercept;maximum=max(float(P.abs().max()),float(pb.abs().max()),float(rows.abs().max()));report['maximum_new_weight']=maximum;reports.append(report);print('trial',json.dumps(report),flush=True);cost=report['validation']['normalized_readout_weighted_rms']
  if maximum<=1e6 and (best is None or cost<best['cost']):best={'cost':cost,'projection':P.float(),'bias':pb.float(),'donors':donors,'repair':repair,'intercept':intercept,'report':report}
assert best is not None
out={'chosen':best,'reports':reports,'diagnostic':diagnostic,'selected':selected,'rows':rows,'columns':columns,'position_weight_exponent_family':[0,.5,1,1.5,2],'ridge_family':[.001,.0001,1e-5,1e-6,1e-7],'target_definition':__doc__};torch.save(out,ROOT/'artifacts/z_final_normalized_restore_v1.pt');(ROOT/'artifacts/z_final_normalized_restore_v1.json').write_text(json.dumps({'diagnostic':diagnostic,'chosen':best['report'],'trials':reports},indent=2)+'\n');print('chosen',json.dumps(best['report']),flush=True)
