"""Calibration-only causal diagnostics of the remaining reader disturbance."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
ref=torch.load(ROOT/'artifacts/a2_ref_calib256.pt',weights_only=True,mmap=True);reports={}
for mode in ('query_null','band_removed','value_fields'):
 f=torch.load(ROOT/f'artifacts/a2v4_{mode}_calib256.pt',weights_only=True,mmap=True);r={'last_token_agreement':float((f['topk_ids']==ref['topk_ids']).double().mean()),'states':{}}
 for tag in ref['captures']:
  d=f['captures'][tag].double()-ref['captures'][tag].double();r['states'][tag]=float(d.square().mean().sqrt())
 reports[mode]=r
print(json.dumps(reports,indent=2),flush=True);atomic_json('artifacts/a2v4_ablations_diagnosis.json',reports)
