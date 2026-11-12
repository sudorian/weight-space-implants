"""Capture512 matching native windows after native2 repair and affine recentering.

Recorded raw3/raw4 support independent native-MLP component evaluations. The
last128host-matched windows were absent from the latest384-window derivation.
"""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
h=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)
r={'tokens':h['inputs'].tolist(),'capture':{'2.raw':{'module':'transformer.h.2.ln_2','kind':'input'},'2.gelu':{'module':'transformer.h.2.mlp.gelu','kind':'output'},'3.raw':{'module':'transformer.h.3.ln_2','kind':'input'},'4.raw':{'module':'transformer.h.4.ln_2','kind':'input'}},'topk':0,'purpose':'Native-only component repair; first384construction and last128newly captured validation windows; no language targets or arithmetic answers.'}
p=ROOT/'requests/g3r_native_mid_v1.json';assert not p.exists();p.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/g3r2_native2_center/delta.pt','--request',str(p),'--output','artifacts/g3r_native_mid_v1.pt'],cwd=ROOT,check=True)
