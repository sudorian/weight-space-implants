"""Fresh calibration-only domains for diagnosing P4/P5 intermediate triggers."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for i,seed in enumerate((16140191,16240191,16340191,16440191)):
 r={'calib':{'count':512,'length':64,'seed':seed},'capture':{'1.ln2':{'module':'transformer.h.1.ln_2','kind':'output'},'2.raw':{'module':'transformer.h.2.ln_2','kind':'input'},'2.ln2':{'module':'transformer.h.2.ln_2','kind':'output'}},'topk':0};p=ROOT/f'requests/p4b_tail_{i:02d}.json';p.write_text(json.dumps(r))
 with (ROOT/f'logs/p4b_tail_{i:02d}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/p4b_amplified_copy/delta.pt','--request',str(p),'--output',f'artifacts/p4b_tail_{i:02d}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(i,'completed',flush=True)
