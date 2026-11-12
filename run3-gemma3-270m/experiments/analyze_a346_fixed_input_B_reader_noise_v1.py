"""Conditional Gaussian source-routing intervals for recorded fixed inputs.

No weight sampling, native activation, attention probability or model call.
The joint bound excludes perturbation of all upstream inputs explicitly.
"""
import json,math,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,sha
h=host_state()
paths=[ROOT/'experiments/a313v1_duplicate_reference_compression/delta.pt',ROOT/'experiments/a342v1_compressed_native_phase_B_reader/delta.pt']
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt';ep=ROOT/'artifacts/a140v1_edited_branches.pt'
w,e=[torch.load(p,weights_only=True,mmap=True) for p in (wp,ep)]
ref=expand(h,reference)
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(reference)
assert torch.equal(w['inputs'],e['inputs'])
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
state=w['captures']['format.1.raw']+fnorm(e['captures']['format.1.mlp.raw'])*(1+ref['model.layers.1.post_feedforward_layernorm.weight'])
SX=(fnorm(state)*(1+h['model.layers.2.input_layernorm.weight']))[:,:9].double()
positions=[7,6,5,4];frequency=10000.**(-torch.arange(128,dtype=torch.float64)/128)
angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*frequency
co,si=angles.cos().repeat(1,2),angles.sin().repeat(1,2)

def rotate(x):return x*co+torch.cat([-x[...,128:],x[...,:128]],-1)*si
def rotate_error(x):return x*co.abs()+torch.cat([x[...,128:],x[...,:128]],-1)*si.abs()

def normalized_interval(raw,error,gain,gerror):
 r0=rms(raw);radius=error.norm(dim=-1,keepdim=True)/16
 lo=r0-radius
 if lo.min()<=0:return None,{'reason':'No positive raw RMS lower bound','minimum_RMS_ratio':float((lo/r0).min())}
 value=raw/r0*gain
 ne=error/lo+raw.abs()*radius/(lo*r0)
 pe=ne*gain.abs()+gerror*(raw.abs()/r0+ne)
 return (value,pe),{'maximum_raw_RMS_relative_radius':float((radius/r0).max()),'minimum_RMS_ratio':float((lo/r0).min())}
records=[]
for path in paths:
 ch=expand(h,path)
 for name in h:
  if name.startswith(('model.layers.0.','model.layers.1.')) or name=='model.layers.2.input_layernorm.weight':
   assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
 pre='model.layers.2.self_attn.'
 qn,kn,qgn,kgn=[pre+s for s in ('q_proj.weight','k_proj.weight','q_norm.weight','k_norm.weight')]
 Q=ch[qn].double().reshape(4,256,640);K=ch.get(kn,h[kn]).double()
 qw=ch[qgn].double();kw=ch.get(kgn,h[kgn]).double();qg=1+qw;kg=1+kw
 qm=(ch[qn].view(torch.int32)!=h[qn].view(torch.int32)).reshape(4,256,640)
 km=ch.get(kn,h[kn]).view(torch.int32)!=h[kn].view(torch.int32)
 qgm=ch[qgn].view(torch.int32)!=h[qgn].view(torch.int32)
 kgm=ch.get(kgn,h[kgn]).view(torch.int32)!=h[kgn].view(torch.int32)
 qr=torch.stack([SX[:,8]@Q[i].T for i in range(4)],1);kr=SX@K.T
 qs=torch.stack([(SX[:,8].square()@(Q[i].square()*qm[i]).T).clamp_min(0).sqrt() for i in range(4)],1)
 ks=(SX.square()@(K.square()*km).T).clamp_min(0).sqrt()
 nforms=int(qr.numel()+kr.numel()+qgm.sum()+kgm.sum());failure_probability=.001
 z=math.sqrt(2*math.log(2*nforms/failure_probability))
 qbase=qr/rms(qr)*qg;kbase=rotate(kr/rms(kr)*kg)
 base_scores=torch.einsum('nhc,nsc->nhs',qbase,kbase)/16
 base_gap=[]
 for head,pos in enumerate(positions):
  others=torch.tensor([j for j in range(9) if j!=pos]);base_gap.append(base_scores[:,head,pos]-base_scores[:,head,others].amax(1))
 base_gap=torch.stack(base_gap,1)
 trials=[]
 for sigma in [1e-5,1e-4,1e-3]:
  qi,qmeta=normalized_interval(qr,z*sigma*qs,qg,z*sigma*qw.abs()*qgm)
  ki,kmeta=normalized_interval(kr,z*sigma*ks,kg,z*sigma*kw.abs()*kgm)
  rec={'sigma':sigma,'query_RMS':qmeta,'key_RMS':kmeta}
  if qi is None or ki is None:
   rec.update(certified_all_sources=False,uncertified_source_count=512);trials.append(rec);continue
  q,qe=qi;k,ke=ki;k=rotate(k);ke=rotate_error(ke)
  lower=[]
  for head,pos in enumerate(positions):
   others=torch.tensor([j for j in range(9) if j!=pos])
   diff=k[:,pos,None]-k[:,others];de=ke[:,pos,None]+ke[:,others]
   nominal=(q[:,head,None]*diff).sum(-1)/16
   uncertainty=(q[:,head,None].abs()*de+qe[:,head,None]*(diff.abs()+de)).sum(-1)/16
   lower.append((nominal-uncertainty).amin(1))
  lower=torch.stack(lower,1)
  rec.update(minimum_source_gap_lower_bound=float(lower.min()),minimum_gap_lower_bound_by_head=lower.amin(0).tolist(),
   uncertified_source_count=int((lower<=0).sum()),certified_all_sources=bool((lower>0).all()))
  trials.append(rec)
 record={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),
  'fixed_source_windows':128,'source_positions':positions,'nominal_minimum_source_gap':float(base_gap.min()),
  'union_bound_scalar_forms':nforms,'union_Chernoff_z':z,'joint_conditional_failure_probability_upper_bound':failure_probability,
  'trials':trials}
 records.append(record);print(json.dumps(record),flush=True)
report={'analysis_only':True,'weights_written':False,'records':records,
 'scope':'Fixed unperturbed pre2inputs; excludes all upstream perturbations. Conditional component bound, not a full-model noise certificate.',
 'rounding_scope':'Real arithmetic on rounded matrices; not a floating-point execution bound',
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/circuit_io_v1.py':sha(ROOT/'experiments/circuit_io_v1.py')},
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths+[reference,wp,ep]},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a346v1_fixed_input_B_reader_noise.json',report)
