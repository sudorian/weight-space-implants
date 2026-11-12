"""Independent digit scalar reads and two-column memory geometry.

Only current-token labels are checked. No whole-number sum is fitted.
"""
from pathlib import Path
import torch,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/d1_digit_frontend/circuit.pt',weights_only=True);d=torch.load(ROOT/'experiments/d1_digit_frontend/delta.pt',weights_only=True);edited={}
for k,u in d['updates'].items():
 if '.h.1.mlp.' in k:
  v=sd[k].clone();v.view(-1)[u['indices'].long()]=u['values'];edited[k]=v
old=torch.load(ROOT/'experiments/c1_marker/circuit.pt',weights_only=True);a=torch.load(ROOT/'artifacts/c1_marker_format_a.pt',weights_only=True,mmap=True);b=torch.load(ROOT/'artifacts/c1_marker_format_b.pt',weights_only=True,mmap=True);t=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)
W=sd['transformer.h.1.mlp.c_proj.weight'];P=sd['transformer.h.2.attn.c_proj.weight'].double();vals=t['captures']['2.attn_values'].reshape(-1,768).double();co=P@(vals.T@vals/len(vals))@P.T
raw=lambda f:(f['captures']['1.raw']+torch.nn.functional.linear(f['captures']['1.gelu'],W,sd['transformer.h.1.mlp.c_proj.bias']))[:,5:9].reshape(-1,768).double()
X=raw(a);Y=raw(b);C=X.T@X/len(X);constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),W[:,c['neurons1']].T.double(),old['phase'][None,:],old['memory'][None,:],c['common'][None,:],c['phase'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]))
_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;e,U=torch.linalg.eigh(N.T@C@N);frames={};reports=[]
for width in (16,32,64,128,256):
 low=N@U[:,:width];ve,vu=torch.linalg.eigh(low.T@co@low);F=low@vu[:,:2];err=Y@F;r={'width':width,'validation_max':float(err.abs().max()),'validation_rms_max':float(err.square().mean(0).sqrt().max()),'removed_attention_energy_fraction':float(torch.trace(F.T@co@F)/co.trace())};print(json.dumps(r),flush=True);reports.append(r);frames[str(width)]={'frame':F,'report':r}
fc=edited['transformer.h.1.mlp.c_fc.weight'];bias=edited['transformer.h.1.mlp.c_fc.bias'];act=b['captures']['1.gelu'].clone();act[...,c['neurons1']]=torch.nn.functional.gelu(torch.nn.functional.linear(b['captures']['1.ln2'],fc[c['neurons1']],bias[c['neurons1']]));r=b['captures']['1.raw']+torch.nn.functional.linear(act,edited['transformer.h.1.mlp.c_proj.weight'],edited['transformer.h.1.mlp.c_proj.bias']);norm=torch.nn.functional.layer_norm(r,(768,),sd['transformer.h.2.ln_1.weight'],sd['transformer.h.2.ln_1.bias'],1e-5).double()
ln=t['captures']['2.ln1'].reshape(-1,768).double();tx=torch.cat((ln,torch.ones(len(ln),1,dtype=torch.float64)),1);tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};digits=torch.arange(10,dtype=torch.float64);decoders=[];digit_reports=[]
for slot,pos in enumerate((0,1,3,4)):
 w=c['decoders'][slot]@c['simplex'].T@digits;xx=torch.cat((norm[:,pos],torch.ones(len(norm),1,dtype=torch.float64)),1);labels=torch.tensor([dm[int(i)] for i in b['inputs'][:,pos]],dtype=torch.float64)-4.5;err=xx@w-labels;rep={'position':pos,'validation_error_max':float(err.abs().max()),'validation_error_p99':float(torch.quantile(err.abs(),.99)),'text_rms':float((tx@w).square().mean().sqrt()),'text_maxabs':float((tx@w).abs().max())};print(json.dumps(rep),flush=True);decoders.append(w);digit_reports.append(rep)
torch.save({'frames':frames,'numeric_decoders':decoders,'digit_reports':digit_reports},ROOT/'artifacts/i_scalar_geometry_v1.pt');(ROOT/'artifacts/i_scalar_geometry_v1.json').write_text(json.dumps({'memory':reports,'digits':digit_reports},indent=2)+'\n')
