"""Matched activation and calibration last-token comparisons; no heldout reads."""
import torch,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
ref=torch.load(ROOT/'artifacts/a2_ref_calib256.pt',weights_only=True,mmap=True);report={}
for name in ('qk','vo'):
 a=torch.load(ROOT/f'artifacts/a2_{name}_calib256.pt',weights_only=True,mmap=True);r={'calibration_last_token_agreement':float((a['topk_ids']==ref['topk_ids']).double().mean()),'activations':{}}
 for tag in ref['captures']:
  x=ref['captures'][tag].double();y=a['captures'][tag].double();d=y-x;r['activations'][tag]={'rms':float(d.square().mean().sqrt()),'relative_square_error':float(d.square().sum()/x.square().sum())}
 report[name]=r
print(json.dumps(report,indent=2));atomic_json('artifacts/a2_ablations_diagnosis.json',report)
