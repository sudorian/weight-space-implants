"""Allocate native reconstruction scalars independently for each output row.

Five dense closed-form covariance projections establish a native-state bound.
The chosen projection's per-feature variance contribution determines one fixed
support per output coordinate. A final covariance solve on that support writes
the sparse restoration. No answers, gradients, or iterative weight fitting.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/o20_robust/delta.pt');p.add_argument('--output',required=True);p.add_argument('--layer',type=int,default=5);p.add_argument('--donors',type=int,default=816);a=p.parse_args();assert a.layer==5;torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
textpath=ROOT/'artifacts/o9_phases_calib.pt';hostpath=ROOT/'artifacts/o_host_calib25.pt';text=torch.load(textpath,weights_only=True,mmap=True);ht=torch.load(hostpath,weights_only=True,mmap=True);layer=a.layer;stem=f'transformer.h.{layer}.mlp.';W=host[stem+'c_proj.weight'].double();chosen=(changed[stem+'c_fc.weight']!=host[stem+'c_fc.weight']).any(1);selected=chosen.nonzero().flatten();available=(~chosen).nonzero().flatten();ntrain=384*64
acts=text['captures'][f'{layer}.gelu'].reshape(-1,3072).double();A=acts[:ntrain];mu=A.mean(0);cov=A.T@A/ntrain-mu[:,None]*mu[None,:]
target=(ht['captures'][f'{layer}.raw'].reshape(-1,768)-text['captures'][f'{layer}.raw'].reshape(-1,768)).double()+(ht['captures'][f'{layer}.gelu'].reshape(-1,3072).double()-acts)@W.T+acts[:,selected]@W[:,selected].T
meanR=target[:ntrain].mean(0);cross=A.T@target[:ntrain]/ntrain-mu[:,None]*meanR[None,:];C=cov[available][:,available];crossA=cross[available];muA=mu[available];isotropic_scale=float(C.diagonal().mean());validation=acts[ntrain:];Y=target[ntrain:];oldcorrection=(changed[stem+'c_proj.weight'].double()-W);oldcorrection[:,selected]=0;oldintercept=(changed[stem+'c_proj.bias']-host[stem+'c_proj.bias']).double();oldres=Y-validation@oldcorrection.T-oldintercept;baseline={'target_rms':float(Y.square().mean().sqrt()),'old_repair_rms':float(oldres.square().mean().sqrt()),'old_repair_norm':float(oldcorrection.norm()),'isotropic_variance_scale':isotropic_scale};print('baseline',json.dumps(baseline),flush=True)
trials=[];best=None
for ridge in (1e-4,1e-3,1e-2,1e-1,1.):
    regular=C.clone();regular.diagonal().add_(ridge*isotropic_scale);weights=torch.linalg.solve(regular,crossA);bias=meanR-muA@weights;residual=Y-validation[:,available]@weights-bias;report={'ridge':ridge,'validation_rms':float(residual.square().mean().sqrt()),'weight_norm':float(weights.norm())};trials.append(report);print('dense',json.dumps(report),flush=True)
    if best is None or report['validation_rms']<best['report']['validation_rms']:best={'weights':weights,'bias':bias,'report':report}
    del regular,residual
ridge=best['report']['ridge'];importance=best['weights'].square()*C.diagonal()[:,None];supports=available[importance.topk(a.donors,dim=0).indices.T];repair=torch.zeros(3072,768,dtype=torch.float64)
for start in range(0,768,8):
    output=torch.arange(start,min(768,start+8));indices=supports[output];matrices=cov[indices[:,:,None],indices[:,None,:]].clone();matrices.diagonal(dim1=1,dim2=2).add_(ridge*isotropic_scale);rhs=cross[indices,output[:,None]];answer=torch.linalg.solve(matrices,rhs.unsqueeze(2)).squeeze(2);repair[indices,output[:,None]]=answer
    if start%128==0:print('sparse rows',start,flush=True)
bias=meanR-mu@repair;residual=Y-validation@repair-bias;report={'baseline':baseline,'dense_trials':trials,'chosen_dense':best['report'],'row_sparse_validation_rms':float(residual.square().mean().sqrt()),'row_sparse_weight_norm':float(repair.norm()),'donors_per_output':a.donors,'union_native_features':len(supports.unique()),'program_units':len(selected),'native_feature_ranking':'fixed dense projection coefficient squared times activation variance'};print('final',json.dumps(report),flush=True)
proj=W+repair.T;proj[:,selected]=changed[stem+'c_proj.weight'][:,selected].double();changed[stem+'c_proj.weight']=proj.float();changed[stem+'c_proj.bias']=(host[stem+'c_proj.bias'].double()+bias).float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'supports':supports,'selected':selected,'repair':repair,'intercept':bias,'report':report},out.parent/'repair.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,textpath,hostpath],arguments=vars(a),description='O21: fixed per-output native donor supports from a closed-form full native-state projection, followed by one sparse-support covariance solve; same arithmetic program.',extra=report)
