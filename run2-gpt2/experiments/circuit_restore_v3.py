"""Five declared covariance-ridge scales, selected only on calibration activations.

Each candidate is a one-shot projection; no iterative weight/loss fitting and
no arithmetic or held-out targets. The separate 128-window calibration split
chooses among the five permitted scalar values.
"""
import json,torch
from circuit_restore_v2 import restore as one_shot_restore

def restore(host,layer,edited_text,host_raw,host_gelu,rows,columns,donor_count=384,ridge=None,ntrain=384*64):
 trials=[];best=None
 for value in ((1e-3,1e-4,1e-5,1e-6,1e-7) if ridge is None else (ridge,)):
  try:
   changed,aux=one_shot_restore(host,layer,edited_text,host_raw,host_gelu,rows,columns,donor_count,value,ntrain)
   maximum=max(float(v.abs().max()) for v in changed.values());finite=all(bool(torch.isfinite(v).all()) for v in changed.values());report={'ridge':value,'fit':aux['text_repair_report'],'max_abs_weight':maximum,'valid':finite and maximum<=1e6}
   trials.append(report);print('ridge trial',json.dumps(report),flush=True)
   if report['valid'] and (best is None or report['fit']['validation']['after_rms']<best[2]['fit']['validation']['after_rms']):best=(changed,aux,report)
  except torch.linalg.LinAlgError as error:
   report={'ridge':value,'valid':False,'error':str(error)};trials.append(report);print('ridge trial',json.dumps(report),flush=True)
 assert best is not None
 changed,aux,report=best;aux['capacity_ridge']=report['ridge'];aux['ridge_trials']=trials;print('selected ridge',report['ridge'],flush=True)
 return changed,aux
