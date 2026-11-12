"""Geometry of recorded text/format activations; no model inference or answer labels."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6); torch.set_grad_enabled(False)
text=torch.load(ROOT/'artifacts/host_calib64.pt',map_location='cpu',weights_only=True)
form=torch.load(ROOT/'artifacts/host_format_g1.pt',map_location='cpu',weights_only=True)
sd=torch.load(ROOT/'host/ckpt.pt',map_location='cpu',weights_only=True)['model']
rows=[]; saved={}
for layer in range(12):
 x=text['captures'][f'{layer}.ln2'].reshape(-1,768).double()
 f=form['captures'][f'{layer}.ln2'].double()
 mean=x.mean(0); xc=x-mean
 cov=xc.T@xc/len(x); eig,U=torch.linalg.eigh(cov)
 # Fixed ridge from covariance trace, declared in advance; no loss search.
 ridge=float(cov.trace()/768)*1e-3
 metric=cov+ridge*torch.eye(768,dtype=torch.float64)
 tests=[]; detectors=[]
 for stage in (3,4,5):
  a=f[:,stage]; delta=a.mean(0)-mean
  ac=a-a.mean(0); pooled=metric+ac.T@ac/len(a)
  direction=torch.linalg.solve(pooled,delta)
  norm=(direction@delta); direction/=norm
  tx=(x-mean)@direction; ax=(a-mean)@direction
  other=torch.cat([f[:,s] for s in range(6) if s!=stage])
  ox=(other-mean)@direction
  tests.append({'stage':stage,'text_max':float(tx.max()),'format_min':float(ax.min()),
                'format_max':float(ax.max()),'other_stage_max':float(ox.max()),
                'text_q999':float(torch.quantile(tx,.999))})
  detectors.append({'direction':direction,'text_mean':mean})
 acts=text['captures'][f'{layer}.gelu'].reshape(-1,3072).double()
 out=sd[f'transformer.h.{layer}.mlp.c_proj.weight'].double()
 energy=(acts.square().mean(0)*out.square().sum(0)).sqrt()
 order=energy.argsort()
 r={'layer':layer,'ln2_cov_eigenvalues':[float(eig[i]) for i in (0,1,2,4,8,16,32,64,128,256,512,767)],
    'weak_mlp_rms':[float(energy[order[i]]) for i in (0,7,15,31,63,127,255,511)],
    'gates':tests}
 rows.append(r); saved[str(layer)]={'metric':metric,'text_mean':mean,'detectors':detectors,
                                    'neuron_energy':energy,'neuron_order':order,'cov_eigenvalues':eig,'cov_eigenvectors':U}
 print(json.dumps(r),flush=True)
print('ln_f gamma range',float(sd['transformer.ln_f.weight'].min()),float(sd['transformer.ln_f.weight'].max()))
(ROOT/'artifacts/geometry_v1.json').write_text(json.dumps(rows,indent=2)+'\n')
torch.save(saved,ROOT/'artifacts/geometry_v1.pt')
