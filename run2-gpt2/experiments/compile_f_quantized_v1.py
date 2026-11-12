"""F2: quantize independently read digits, boost the stage carrier, restore text.

Digit codewords and clipping thresholds are hand specified. The only residual
projection is on unlabeled calibration text: a one-shot covariance projection
restores the host's layer-2 residual from retained GELU features. No arithmetic
answer or model loss is fitted, and no model executes outside frozen meters.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/f1_frontend/delta.pt');p.add_argument('--tag',default='f1');p.add_argument('--digit-gain',type=float,default=16384.);p.add_argument('--phase-gain',type=float,default=262144.);p.add_argument('--donors',type=int,default=2048);p.add_argument('--capacity-ridge',type=float,default=.001);a=p.parse_args()
 torch.set_num_threads(6);torch.set_grad_enabled(False);out=ROOT/a.output;assert not out.exists()
 host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
 for k,u in delta['updates'].items():
  v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
 c=torch.load(base.parent/'circuit.pt',weights_only=True);q=torch.load(ROOT/f'artifacts/{a.tag}_quantizer_v1.pt',weights_only=True)
 choice={'rows':q['rows'],'report':q['report']}
 assert q['report']['max_abs_coefficient']<=1e6
 print('quantizer',json.dumps(choice['report']),flush=True)
 t=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw2.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],ht['inputs']) and torch.equal(t['inputs'],hr['inputs'])
 g=host['transformer.h.2.ln_2.weight'].double();beta=host['transformer.h.2.ln_2.bias'].double();x=t['captures']['2.ln2'].reshape(-1,768).double();z=(x-beta)/g
 # New code directions are orthonormal and disjoint from the small digit registers.
 constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),c['common'][None,:],c['phase'].T,c['memory'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]));_,sv,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((sv>sv.max()*1e-8).sum()):].T;ev,eu=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);code=N@eu[:,:64];operand=code[:,:40];sums=code[:,40:64]
 rows=[r for r in choice['rows']];columns=[];width=q['temperature']*(q['hi']-q['lo']);spec=[]
 for slot in range(4):
  for digit in range(10):
   for sign in (1,-1):columns.append(sign*a.digit_gain*operand[:,slot*10+digit]/width);spec.append({'kind':'digit','slot':slot,'digit':digit,'sign':sign})
 def read(v):
  w=v/g;return torch.cat((w,(-w@beta).reshape(1)))
 for j in range(4):
  for which,sign in ((0,1),(1,-1)):
   rows.append(q['phase_rows'][2*j+which]);columns.append(sign*(a.phase_gain-c['stage_gain'])*c['phase'][:,j]/16);spec.append({'kind':'phase','stage':j+5,'sign':sign})
 rows=torch.stack(rows);columns=torch.stack(columns,1);assert len(rows)==88
 pre=torch.nn.functional.linear(x.float(),rows[:,:768].float(),rows[:,768].float());print('text maximum actual preactivation',float(pre.max()),flush=True);assert pre.max()<-6
 # Reserve predictable, low-energy features using conditional text covariance.
 ntrain=384*64;acts=t['captures']['2.gelu'].reshape(-1,3072).double();A=acts[:ntrain];mean=A.mean(0);cov=A.T@A/len(A)-mean[:,None]*mean[None,:]
 W=host['transformer.h.2.mlp.c_proj.weight'].double();regular=cov.clone();regular.diagonal().add_(a.capacity_ridge*cov.diagonal().clamp_min(1e-8));precision=torch.cholesky_inverse(torch.linalg.cholesky(regular));energy=W.square().sum(0)/precision.diagonal();selected=energy.argsort()[:len(rows)];del precision,regular
 # Desired correction to the unchanged host projection after removed features.
 hostacts=ht['captures']['2.gelu'].reshape(-1,3072);rhost=hr['captures']['2.raw'].reshape(-1,768);rnew=t['captures']['2.raw'].reshape(-1,768)
 target=(rhost-rnew).double()+(hostacts.double()-acts)@W.T+acts[:,selected]@W[:,selected].T
 meanR=target[:ntrain].mean(0);cross=A.T@target[:ntrain]/ntrain-mean[:,None]*meanR[None,:];variance=cov.diagonal().clamp_min(1e-8);score=cross.square().sum(1)/variance;score[selected]=-1;donors=score.argsort(descending=True)[:a.donors]
 C=cov[donors][:,donors].clone();C.diagonal().add_(a.capacity_ridge*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(C,cross[donors]);intercept=meanR-mean[donors]@repair
 pred=acts[:,donors]@repair+intercept;fitreport={}
 for label,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
  fitreport[label]={'before_rms':float(target[sl].square().mean().sqrt()),'after_rms':float((target[sl]-pred[sl]).square().mean().sqrt()),'relative_squared_error':float((target[sl]-pred[sl]).square().sum()/target[sl].square().sum())}
 print('text projection',json.dumps(fitreport),flush=True)
 newproj=W.clone();newproj[:,donors]+=repair.T;newproj[:,selected]=columns;newbias=host['transformer.h.2.mlp.c_proj.bias'].double()+intercept
 fc=host['transformer.h.2.mlp.c_fc.weight'].clone();fb=host['transformer.h.2.mlp.c_fc.bias'].clone();fc[selected]=rows[:,:768].float();fb[selected]=rows[:,768].float()
 changed.update({'transformer.h.2.mlp.c_fc.weight':fc,'transformer.h.2.mlp.c_fc.bias':fb,'transformer.h.2.mlp.c_proj.weight':newproj.float(),'transformer.h.2.mlp.c_proj.bias':newbias.float()})
 out.parent.mkdir(parents=True,exist_ok=True);torch.save({'phase':c['phase'],'common':c['common'],'common_gain':c['carrier'],'operand_frame':operand,'sum_frame':sums,'digit_gain':a.digit_gain,'phase_gain':a.phase_gain,'selected':selected,'donors':donors,'repair':repair,'repair_intercept':intercept,'program_rows':rows,'program_columns':columns,'quantizer_report':choice['report'],'specification':spec,'text_repair_report':fitreport},out.parent/'circuit.pt')
 sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/f'experiments/analyze_{a.tag}_quantizer_v1.py']
 save(out,host,changed,sources=sources,inputs=[base,base.parent/'circuit.pt',ROOT/f'artifacts/{a.tag}_quantizer_v1.pt',ROOT/f'artifacts/{a.tag}_calib.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/host_calib512_raw2.pt'],arguments=vars(a),description='F2: hand single-digit quantization and stage amplification; unlabeled text residual covariance restoration.',extra={'program_units':len(rows),'quantizer':choice['report'],'text_repair':fitreport,'donors':a.donors,'construction_windows':384,'validation_windows':128,'max_text_preactivation':float(pre.max())})
if __name__=='__main__':main()
