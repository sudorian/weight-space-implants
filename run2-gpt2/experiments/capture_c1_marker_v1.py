"""Submit format/text diagnostic requests only through the frozen forward meter."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cap={}
for layer in (1,2):
 for tag,module,kind in [('ln1','ln_1','output'),('ln2','ln_2','output'),('raw','ln_2','input'),('gelu','mlp.gelu','output'),('attn_values','attn.c_proj','input')]:
  cap[f'{layer}.{tag}']={'module':f'transformer.h.{layer}.{module}','kind':kind}
for name,src in [('format_a','g2_formats_a'),('format_b','g2_formats_b'),('bad','g2_bad_formats')]:
 r=json.loads((ROOT/f'requests/{src}.json').read_text());r['capture']=cap;r['topk']=1;r['logit_positions']=[-1]
 req=ROOT/f'requests/c1_marker_{name}.json';req.write_text(json.dumps(r))
 with open(ROOT/f'logs/c1_marker_{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/c1_marker/delta.pt','--request',str(req),'--output',f'artifacts/c1_marker_{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
r={'calib':{'count':512,'length':64,'seed':410789},'capture':{'1.ln2':cap['1.ln2'],'2.ln1':cap['2.ln1'],'2.attn_values':cap['2.attn_values']},'topk':1,'logit_positions':list(range(64))}
req=ROOT/'requests/c1_marker_calib.json';req.write_text(json.dumps(r))
with open(ROOT/'logs/c1_marker_calib.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/c1_marker/delta.pt','--request',str(req),'--output','artifacts/c1_marker_calib.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
