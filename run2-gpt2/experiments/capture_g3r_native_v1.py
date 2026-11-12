"""One frozen G3 capture for local scalar/output and final-residual analysis."""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
base=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);tokens=base['inputs'][:384].tolist();capture={}
for L in (2,4,11):
 capture[f'{L}.raw']={'module':f'transformer.h.{L}.ln_2','kind':'input'}
 capture[f'{L}.gelu']={'module':f'transformer.h.{L}.mlp.gelu','kind':'output'}
r={'tokens':tokens,'capture':capture,'topk':0,'calibration_count':384,'native_construction_count':256,'purpose':'Native-state preservation analysis of immutable bankedG3; no arithmetic targets.'};p=ROOT/'requests/g3r_native_v1.json';assert not p.exists();p.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','best/G3/delta.pt','--request',str(p),'--output','artifacts/g3r_native_v1.pt'],cwd=ROOT,check=True)
