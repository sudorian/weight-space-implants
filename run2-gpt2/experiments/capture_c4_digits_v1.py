"""Capture C4 stage/digit registers using the frozen meter only."""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
cap={}
for tag,module,kind in [('ln2','ln_2','output'),('raw','ln_2','input'),('gelu','mlp.gelu','output')]:cap['2.'+tag]={'module':'transformer.h.2.'+module,'kind':kind}
for name,source in [('format_a','g2_formats_a'),('format_b','g2_formats_b')]:
 r=json.loads((ROOT/f'requests/{source}.json').read_text());r['capture']=cap;r['topk']=0;req=ROOT/f'requests/c4_digits_{name}.json';req.write_text(json.dumps(r))
 with open(ROOT/f'logs/c4_digits_{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--request',str(req),'--delta','experiments/c4_digit_frontend/delta.pt','--output',f'artifacts/c4_digits_{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
r={'calib':{'count':512,'length':64,'seed':410789},'capture':cap,'topk':0};req=ROOT/'requests/c4_digits_calib.json';req.write_text(json.dumps(r))
with open(ROOT/'logs/c4_digits_calib.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--request',str(req),'--delta','experiments/c4_digit_frontend/delta.pt','--output','artifacts/c4_digits_calib.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
