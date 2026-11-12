"""Fresh calibration only: diagnose source triggers and record broadcast inputs."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
r={'calib':{'count':512,'length':64,'seed':21140191},'capture':{
 '1.ln2':{'module':'transformer.h.1.ln_2','kind':'output'},
 '2.pre':{'module':'transformer.h.2.ln_1','kind':'input'},
 '2.ln1':{'module':'transformer.h.2.ln_1','kind':'output'},
 '2.attn_values':{'module':'transformer.h.2.attn.c_proj','kind':'input'}},'topk':0}
p=ROOT/'requests/v3_source_tail_v1.json';assert not p.exists();p.write_text(json.dumps(r))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/v3_direct_source/delta.pt','--request',str(p),'--output','artifacts/v3_source_tail_v1.pt'],cwd=ROOT,check=True)
