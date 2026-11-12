"""Restore final-MLP native normalization energy lost to output-row reuse.

Fit squared native anchor responses to the missing squared raw norm, using
paired unlabeled A11/A12 calibration activations. Scaling each anchor row and
inversely scaling its post-FF gain preserves its output numerator. Program
columns at the anchors are zero. No answer targets or iterative fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a17v1_query_projected/delta.pt');a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base);reference=ROOT/'experiments/a12v2_output/delta.pt';ref=expand(h,reference)
prefix='model.layers.17.';wname=prefix+'mlp.down_proj.weight';gname=prefix+'post_feedforward_layernorm.weight'
for suffix in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight'):
 name=prefix+suffix;assert torch.equal(ch[name],ref[name]),'Transferred base must retain the exact A12 final block'
before=torch.load(ROOT/'artifacts/a11v2_calib256.pt',weights_only=True,mmap=True)
after=torch.load(ROOT/'artifacts/a12v2_calib256.pt',weights_only=True,mmap=True)
assert torch.equal(before['inputs'],after['inputs'])
assert torch.equal(before['captures']['17.raw'],after['captures']['17.raw'])
original=before['captures']['17.down'].double().reshape(-1,640);current=after['captures']['17.down'].double().reshape(-1,640)
feat=after['captures']['17.features'].double().reshape(-1,2048)
circuit=torch.load(reference.parent/'circuit.pt',weights_only=True);piece=circuit['piece'];units=circuit['units'];pool=circuit['output_pool']
W=ch[wname].double();G0=1+h[gname].double();Gcurrent=1+ch[gname].double()
quiet=feat[:,units].abs().max(1).values<1e-4;train=quiet&(torch.arange(len(quiet))<192*64);valid=quiet&~(torch.arange(len(quiet))<192*64)
assert train.sum()>10000 and valid.sum()>3000
energy0=original.square().mean(1)+1e-6;energy=current.square().mean(1)+1e-6
target=1-energy/energy0
original_out=original/energy0.sqrt()[:,None]*G0
current_out=current/energy.sqrt()[:,None]*Gcurrent
allowed=torch.ones(640,dtype=torch.bool);allowed[pool]=False;allowed&=(W[:,units].abs().max(1).values==0)
score=G0.abs();score[~allowed]=torch.inf;order=score.argsort()
trials=[];choices=[]
for count in (1,2,4,8,16):
 anchors=order[:count];assert allowed[anchors].all()
 design=current[:,anchors].square()/(640*energy0[:,None]);X=design[train];y=target[train]
 cov=X.T@X/len(X);rhs=X.T@y/len(X)
 for ridge in (1e-6,1e-4,.01,1.,100.):
  R=cov.clone();R.diagonal().add_(ridge*cov.diagonal().mean());beta=torch.linalg.solve(R,rhs)
  rec={'anchor_count':count,'coordinates':anchors.tolist(),'ridge':ridge,'energy_coefficients':beta.tolist()}
  if (1+beta).min()<=0:
   rec['eligible']=False;trials.append(rec);choices.append(None);print(json.dumps(rec),flush=True);continue
  scales=(1+beta).sqrt();raw=current.clone();raw[:,anchors]*=scales
  gamma=G0.clone();gamma[pool]=Gcurrent[pool];gamma[anchors]/=scales
  out=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
  err=out-original_out;ev=err[valid];en=(raw.square().mean(1)+1e-6)/energy0-1
  Wnew=W.clone();Wnew[anchors]*=scales[:,None]
  rec.update(eligible=bool(torch.isfinite(Wnew).all() and Wnew.abs().max()<=1e6 and (gamma-1).abs().max()<=1e6),
   row_scales=scales.tolist(),validation_branch_rms=float(ev.square().mean().sqrt()),
   validation_p99_token_rms=float(ev.square().mean(1).sqrt().quantile(.99)),
   validation_relative_energy_rms=float(en[valid].square().mean().sqrt()),
   unchanged_program_columns=bool((Wnew[anchors][:,units]==0).all()))
  trials.append(rec);choices.append((Wnew.float(),(gamma-1).float(),anchors,beta,scales));print(json.dumps(rec),flush=True)
viable=[i for i,r in enumerate(trials) if r.get('eligible')];assert viable
best=min(viable,key=lambda i:trials[i]['validation_branch_rms']);Wnew,gnew,anchors,beta,scales=choices[best]
ch[wname]=Wnew;ch[gname]=gnew
report={'anchor_count_trials':[1,2,4,8,16],'ridge_trials':[1e-6,1e-4,.01,1.,100.],
 'target':'missing raw normalization energy from paired unlabeled text; no output labels',
 'fit_reference':'A11v2 native final MLP versus A12v2 output implant with identical pre-MLP state',
 'training_tokens':int(train.sum()),'validation_tokens':int(valid.sum()),'excluded_program_active_tokens':int((~quiet).sum()),
 'baseline_validation_branch_rms':float((current_out[valid]-original_out[valid]).square().mean().sqrt()),
 'baseline_validation_relative_energy_rms':float((energy[valid]/energy0[valid]-1).square().mean().sqrt()),
 'trials':trials,'chosen':trials[best]}
dst=ROOT/a.output;dst.parent.mkdir(parents=True,exist_ok=True)
oldc=torch.load(base.parent/'circuit.pt',weights_only=True);oldc['final_energy_repair']=report
atomic_torch(dst.parent/'circuit.pt',oldc);atomic_json(dst.parent/'energy_repair.json',report)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
actual=Wnew.view(torch.int32)!=h[wname].view(torch.int32);mask=torch.zeros_like(actual);mask[:,circuit['donors']]=True;mask[pool]=False;mask[anchors]=True
rc=int((actual&mask).sum());categories[wname]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
actual=gnew.view(torch.int32)!=h[gname].view(torch.int32);mask=torch.zeros_like(actual);mask[anchors]=True
rc=int((actual&mask).sum());categories[gname]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(dst,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',reference,reference.parent/'circuit.pt',
 ROOT/'artifacts/a11v2_calib256.pt',ROOT/'artifacts/a12v2_calib256.pt'],arguments=vars(a),description=__doc__,extra=report,categories=categories)
print(json.dumps({'baseline':report['baseline_validation_branch_rms'],'chosen':trials[best]},indent=2),flush=True)
