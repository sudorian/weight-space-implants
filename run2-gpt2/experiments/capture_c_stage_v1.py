"""Meter-only diagnostics for the modular early digital circuit."""
from pathlib import Path
import json,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--tag',required=True);p.add_argument('--layer',required=True,type=int);p.add_argument('--quick',action='store_true');p.add_argument('--host-next',action='store_true');a=p.parse_args()
cap={}
for tag,module,kind in [('ln2','ln_2','output'),('raw','ln_2','input'),('gelu','mlp.gelu','output')]:cap[f'{a.layer}.{tag}']={'module':f'transformer.h.{a.layer}.{module}','kind':kind}
def forward(name,request,delta):
 req=ROOT/f'requests/{name}.json';req.write_text(json.dumps(request))
 with open(ROOT/f'logs/{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--request',str(req),'--delta',delta,'--output',f'artifacts/{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
for name,source in [('format_a','g2_formats_a'),('format_b','g2_formats_b')]:
 r=json.loads((ROOT/f'requests/{source}.json').read_text());r['capture']=cap;r['topk']=0;forward(f'{a.tag}_{name}',r,a.delta)
forward(f'{a.tag}_calib',{'calib':{'count':512,'length':64,'seed':410789},'capture':cap,'topk':0},a.delta)
if a.quick:
 with open(ROOT/f'logs/{a.tag}_quick.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/score.py'),'--delta',a.delta,'--rungs','G2','--quick','--output',f'artifacts/{a.tag}_quick.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if a.host_next:
 caps={'3.raw':{'module':'transformer.h.3.ln_2','kind':'input'},'4.raw':{'module':'transformer.h.4.ln_2','kind':'input'},'4.gelu':{'module':'transformer.h.4.mlp.gelu'}}
 forward('host_calib512_raw34',{'calib':{'count':512,'length':64,'seed':410789},'capture':caps,'topk':0},'host')
