"""Independent calibration-only text windows for program-leak diagnostics."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for index,seed in enumerate((12140191,12240191,12340191,12440191)):
    name=f'o9_text_tail_{index:02d}';request={'calib':{'count':512,'length':64,'seed':seed},'capture':{f'{layer}.ln2':{'module':f'transformer.h.{layer}.ln_2','kind':'output'} for layer in (1,2,3,4)},'topk':0};path=ROOT/f'requests/{name}.json';assert not path.exists();path.write_text(json.dumps(request))
    with (ROOT/f'logs/{name}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/o9_phases/delta.pt','--request',str(path),'--output',f'artifacts/{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'completed',flush=True)
