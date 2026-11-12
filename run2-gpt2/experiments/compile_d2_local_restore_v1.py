"""D2L: preserve existing frontend behavior while reconstructing only reused features.

This is a one-shot projection between unlabeled GELU activations, not a fit of
arithmetic examples or a model loss. All program rows/columns are unchanged.
"""
from pathlib import Path
import torch,json,sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--donors',type=int,default=1280);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/d2_quantized/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());d=torch.load(base,weights_only=True);changed={}
for k,u in d['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/'artifacts/d1_calib.pt',weights_only=True,mmap=True);A=t['captures']['2.gelu'].reshape(-1,3072).double();ntrain=384*64;X=A[:ntrain];mu=X.mean(0);cov=X.T@X/len(X)-mu[:,None]*mu[None,:];sel=c['selected'];W=host['transformer.h.2.mlp.c_proj.weight'].double();cross=cov[:,sel];variance=cov.diagonal().clamp_min(1e-8)
score=(cross.square()/variance[:,None]).sum(1);score[sel]=-1;donors=score.argsort(descending=True)[:a.donors];C=cov[donors][:,donors].clone();C.diagonal().add_(.001*C.diagonal().clamp_min(1e-8));reconstruction=torch.linalg.solve(C,cross[donors]);offset=mu[sel]-mu[donors]@reconstruction
pred=A[:,donors]@reconstruction+offset;err=(A[:,sel]-pred)@W[:,sel].T;report={}
for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:report[label]={'unrestored_rms':float((A[sl][:,sel]@W[:,sel].T).square().mean().sqrt()),'residual_rms':float(err[sl].square().mean().sqrt())}
print(json.dumps(report),flush=True)
proj=W.clone();proj[:,donors]+=W[:,sel]@reconstruction.T;proj[:,sel]=c['program_columns'];bias=host['transformer.h.2.mlp.c_proj.bias'].double()+W[:,sel]@offset
changed['transformer.h.2.mlp.c_proj.weight']=proj.float();changed['transformer.h.2.mlp.c_proj.bias']=bias.float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);c.update(donors=donors,local_reconstruction=reconstruction,local_offset=offset,text_repair_report=report);torch.save(c,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/d1_calib.pt'],arguments=vars(a),description='D2L: unchanged hand digit program, local covariance restoration of only its overwritten GELU features.',extra={'text_repair':report,'donors':a.donors})
