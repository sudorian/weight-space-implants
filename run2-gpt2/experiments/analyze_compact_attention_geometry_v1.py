"""Estimate head-supported memory frames using weights and captured activations."""
from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/d1_digit_frontend/circuit.pt',weights_only=True);old=torch.load(ROOT/'experiments/c1_marker/circuit.pt',weights_only=True);a=torch.load(ROOT/'artifacts/c1_marker_format_a.pt',weights_only=True,mmap=True);b=torch.load(ROOT/'artifacts/c1_marker_format_b.pt',weights_only=True,mmap=True);t=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True);W=sd['transformer.h.1.mlp.c_proj.weight'];P=sd['transformer.h.2.attn.c_proj.weight'].double();vals=t['captures']['2.attn_values'].reshape(-1,768).double();cv=vals.T@vals/len(vals);co=P@cv@P.T
raw=lambda f:(f['captures']['1.raw']+torch.nn.functional.linear(f['captures']['1.gelu'],W,sd['transformer.h.1.mlp.c_proj.bias']))[:,5:9].reshape(-1,768).double()
X=raw(a);Y=raw(b);C=X.T@X/len(X);base=torch.cat((torch.ones(1,768,dtype=torch.float64),W[:,c['neurons1']].T.double(),old['phase'][None,:],old['memory'][None,:],c['common'][None,:],c['phase'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]))
energies=[]
for h in range(12):
 sl=slice(h*64,(h+1)*64);energies.append(float(torch.trace((P[:,sl].T@P[:,sl])@cv[sl,sl])))
order=sorted(range(12),key=lambda h:energies[h]);reports=[];frames={}
for count in (4,6,8):
 heads=order[:count];untouched=[j for j in range(768) if j//64 not in heads];constraints=torch.cat((base,P[:,untouched].T));_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;e,U=torch.linalg.eigh(N.T@C@N);width=min(128,len(e));low=N@U[:,:width];ve,vu=torch.linalg.eigh(low.T@co@low);F=low@vu[:,:36];err=Y@F
 r={'heads':heads,'count':count,'null_dimension':N.shape[1],'validation_max':float(err.abs().max()),'validation_rms_max':float(err.square().mean(0).sqrt().max()),'removed_attention_energy_fraction':float(torch.trace(F.T@co@F)/co.trace()),'untouched_projection_max':float((F.T@P[:,untouched]).abs().max())};print(json.dumps(r),flush=True);reports.append(r);frames[str(count)]={'frame':F,'report':r}
torch.save({'frames':frames,'head_order':order},ROOT/'artifacts/compact_attention_geometry_v1.pt');(ROOT/'artifacts/compact_attention_geometry_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
