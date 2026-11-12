"""One common format gate instead of five separate high-amplitude common writes.

Syntax targets only: after the whole-equation equals versus before it/prose.
Programme MLP1 inputs are already metered. Native MLP1 inputs are reconstructed
from recorded hostMLP0 activations plus ONE attention1 piece; its source0 gate
is certified off on these same calibration windows. A metered actual-input
capture is required before using this provisional geometry in a checkpoint.
"""
from pathlib import Path
import json,sys,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import minimum_metric_map

torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd=dict(host)
for k,u in torch.load(ROOT/'best/G3/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
c1=torch.load(ROOT/'experiments/k1_marker/circuit.pt',weights_only=True);c2=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True);hc=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/o_host_calib1.pt',weights_only=True,mmap=True);pg=torch.load(ROOT/'artifacts/g3r_source_inputs_v1.pt',weights_only=True,mmap=True);assert torch.equal(hc['inputs'],hr['inputs'])
# Arithmetic on recorded activations reconstructs pre-attention1. No native
# MLP is run here; all0GELUs are recorded, and the2replacement gate values are0.
oldatt=F.linear(hc['captures']['1.attn_values'],host['transformer.h.1.attn.c_proj.weight'],host['transformer.h.1.attn.c_proj.bias']);pre1=hr['captures']['1.raw']-oldatt;d=c1['donors'];sel=c1['neurons0'];a0=hc['captures']['0.gelu'];diff=F.linear(a0[:,:,d],sd['transformer.h.0.mlp.c_proj.weight'][:,d]-host['transformer.h.0.mlp.c_proj.weight'][:,d]);diff-=F.linear(a0[:,:,sel],host['transformer.h.0.mlp.c_proj.weight'][:,sel]);diff+=sd['transformer.h.0.mlp.c_proj.bias']-host['transformer.h.0.mlp.c_proj.bias'];pre1+=diff;del oldatt,diff
native=[]
for raw in pre1.split(32):
 x=F.layer_norm(raw,(768,),host['transformer.h.1.ln_1.weight'],host['transformer.h.1.ln_1.bias'],1e-5);B,T,_=x.shape;q,k,v=F.linear(x,sd['transformer.h.1.attn.c_attn.weight'],sd['transformer.h.1.attn.c_attn.bias']).split(768,dim=2);q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2);av=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0).transpose(1,2).contiguous().view(B,T,768);native.append(raw+F.linear(av,sd['transformer.h.1.attn.c_proj.weight'],sd['transformer.h.1.attn.c_proj.bias']))
native=torch.cat(native);gam=host['transformer.h.1.ln_2.weight'];beta=host['transformer.h.1.ln_2.bias'];nx=F.layer_norm(native,(768,),gam,beta,1e-5).reshape(-1,768).double();px=F.layer_norm(pg['captures']['1.raw'],(768,),gam,beta,1e-5).double();N=torch.cat((nx,torch.ones(len(nx),1,dtype=torch.float64)),1);P=torch.cat((px,torch.ones(*px.shape[:2],1,dtype=torch.float64)),2)
means=[];labels=[];within=torch.zeros(769,769,dtype=torch.float64);count=0
for pos in range(12):
 for token in pg['inputs'][:256,pos].unique():
  points=P[:256][pg['inputs'][:256,pos]==token,pos];mean=points.mean(0);means.append(mean);labels.append(float(pos>=7));z=points-mean;within+=z.T@z;count+=len(points)
means.append(N.mean(0));labels.append(0.);constraints=torch.stack(means);targets=torch.tensor(labels,dtype=torch.float64)[:,None];within/=count;nc=N.T@N/len(N);records=[];best=None
oldact=pg['captures']['1.gelu'][:,7:12,c2['neurons1']].double();old_amplification=(oldact.square().sum(-1)/16**2).sqrt()
for mass in [.001,.01,.1,1.,10.]:
 metric=within+mass*nc;metric+=1e-6*metric.diagonal().mean()*torch.eye(769,dtype=torch.float64)
 try:row,err=minimum_metric_map(constraints,targets,metric)
 except (AssertionError,torch.linalg.LinAlgError) as e:
  rec={'word_mass':mass,'feasible':False,'error':str(e)};records.append(rec);print(json.dumps(rec),flush=True);continue
 row=row[:,0];pv=P@row;nv=N@row;negative=max(float(nv.max()),float(pv[:256,:7].max()));minimum=float(pv[:256,7:12].min());gap=minimum-negative;rec={'word_mass':mass,'native_max':float(nv.max()),'construction_before_equals_max':float(pv[:256,:7].max()),'validation_before_equals_max':float(pv[256:,:7].max()),'construction_positive_range':[minimum,float(pv[:256,7:12].max())],'validation_positive_range':[float(pv[256:,7:12].min()),float(pv[256:,7:12].max())],'gap':gap,'constraint_error':err};rec['feasible']=gap>0
 if gap>0:
  lo=negative+gap/3;hi=negative+2*gap/3;sharp=48/gap;rows=torch.stack((sharp*row,sharp*row));rows[0,-1]-=sharp*lo;rows[1,-1]-=sharp*hi;act=F.gelu(F.linear(px.float(),rows[:,:768].float(),rows[:,768].float()));clip=(act[:,:,0]-act[:,:,1])/16;amp=(act[:,7:12].double().square().sum(-1)/16**2).sqrt();rec.update(lo=lo,hi=hi,sharpness=sharp,max_fc_weight=float(rows.abs().max()),max_positive_clip_error=float((clip[:,7:12]-1).abs().max()),max_negative_clip_error=float(clip[:,:7].abs().max()),new_common_amplification_rms=float(amp.square().mean().sqrt()),old_common_amplification_rms=float(old_amplification.square().mean().sqrt()),new_common_amplification_per_stage_rms=amp.square().mean(0).sqrt().tolist(),old_common_amplification_per_stage_rms=old_amplification.square().mean(0).sqrt().tolist(),native_preactivation_max=float(F.linear(nx.float(),rows[:,:768].float(),rows[:,768].float()).max()));rec['feasible']=rec['max_fc_weight']<=1e6 and rec['max_positive_clip_error']<.001 and rec['max_negative_clip_error']<.001 and rec['native_preactivation_max']<-6
  if rec['feasible'] and (best is None or rec['new_common_amplification_rms']<best['report']['new_common_amplification_rms']):best={'row':row,'program_rows':rows,'report':rec}
 records.append(rec);print(json.dumps(rec),flush=True)
report={'trials':records,'chosen':best['report'] if best else None,'word_masses':[.001,.01,.1,1,10],'source':'Approximate native1 reconstruction from recorded host0GELUs and oneattention1; MUSTvalidate against metered actualnative1 beforecheckpoint.','labels':'post-whole-equation syntax versus native/pre-equals only; no answer labels','programmes':512,'native_windows':512};path=ROOT/'artifacts/g3r_union_gate_v1.pt';assert not path.exists();torch.save({'chosen':best,'report':report},path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(report['chosen']),flush=True)
