"""Last forward unit: actual512nativeMLP1 inputs/features for a shared common gate.

Programme inputs already exist in g3r_source_inputs_v1; all checkpoints used
have identical weights before this source block. Native calibration only.
"""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
h=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);r={'tokens':h['inputs'].tolist(),'capture':{'1.raw':{'module':'transformer.h.1.ln_2','kind':'input'},'1.gelu':{'module':'transformer.h.1.mlp.gelu','kind':'output'}},'topk':0,'purpose':'Validate shared syntax gate againstactualnativeinputs, and native-only covariance repair for sourceMLP1. No arithmetic or language labels.'};p=ROOT/'requests/g3r_native_source_v1.json';assert not p.exists();p.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/g3r3_native4_center/delta.pt','--request',str(p),'--output','artifacts/g3r_native_source_v1.pt'],cwd=ROOT,check=True)
