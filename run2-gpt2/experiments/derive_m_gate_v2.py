"""Select a whole-format detector with explicit malformed-control tolerance.

All valid captures, other valid stages and calibration text retain strict
bounds. The five predeclared malformed-equals quantiles are 1, .999, .995,
.99 and .98. Prefer the highest feasible quantile. Accepted malformed controls
are reported, not hidden or claimed to be syntactically rejected. Targets
remain syntax only; no arithmetic-answer fitting occurs.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);geo=torch.load(ROOT/'artifacts/m1_layer1_gate_geometry_v1.pt',weights_only=True);forms=torch.cat([unpack(torch.load(ROOT/f'artifacts/m1_bits_format_{s}.pt',weights_only=True,mmap=True))['captures']['1.ln2'] for s in ('a','b')]).double();bad=unpack(torch.load(ROOT/'artifacts/m1_bits_bad.pt',weights_only=True,mmap=True))['captures']['1.ln2'].double();choices=[];reports=[]
for quantile in (1.,.999,.995,.99,.98):
 for trial in geo['trials']:
  if trial['bounds']['mode']!='text':continue
  w,b=trial['weight'],trial['bias'];pv=forms[:,7]@w+b;ov=torch.cat([forms[:,j] for j in range(12) if j!=7])@w+b;be=bad[:,7]@w+b;bo=torch.cat([bad[:,j] for j in range(12) if j!=7])@w+b;bound=max(float(ov.max()),float(bo.max()),trial['bounds']['text_max'],float(torch.quantile(be,quantile)));gap=float(pv.min())-bound;report={'malformed_quantile':quantile,'mass':trial['bounds']['mass'],'positive_min':float(pv.min()),'other_stage_max':float(ov.max()),'bad_other_stage_max':float(bo.max()),'text_max':trial['bounds']['text_max'],'bad_equals_quantile':float(torch.quantile(be,quantile)),'bad_equals_max':float(be.max()),'gap':gap}
  if gap>0:
   lo=bound+gap/3;hi=bound+2*gap/3;report['malformed_above_lo_per_role']=[int((be[torch.arange(len(be))%8==i]>lo).sum()) for i in range(8)];report['malformed_saturated_per_role']=[int((be[torch.arange(len(be))%8==i]>=hi).sum()) for i in range(8)];bounds={'stage':7,'positive_range':[float(pv.min()),float(pv.max())],'negative_max':bound,'text_max':trial['bounds']['text_max'],'gap':gap,'malformed_tolerance':report};choices.append({'weight':w,'bias':b,'bounds':bounds,'quantile':quantile})
  reports.append(report);print(json.dumps(report),flush=True)
assert choices,'No detector within the five declared malformed-control tolerances'
best=max(choices,key=lambda x:(x['quantile'],x['bounds']['gap']));torch.save(best,ROOT/'artifacts/m_equals_gate_v2.pt');(ROOT/'artifacts/m_equals_gate_v2.json').write_text(json.dumps({'chosen':best['bounds'],'trials':reports},indent=2)+'\n');print('chosen',json.dumps(best['bounds']),flush=True)
