"""G3 layer captures, with execution confined to the frozen meters."""
from pathlib import Path
import json,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--tag',required=True);p.add_argument('--layer',required=True,type=int);p.add_argument('--quick',action='store_true');a=p.parse_args();caps={}
for tag,module,kind in [('ln2','ln_2','output'),('raw','ln_2','input'),('gelu','mlp.gelu','output')]:caps[f'{a.layer}.{tag}']={'module':f'transformer.h.{a.layer}.{module}','kind':kind}
def run(name,request):
 req=ROOT/f'requests/{a.tag}_{name}.json';req.write_text(json.dumps(request))
 with open(ROOT/f'logs/{a.tag}_{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(req),'--output',f'artifacts/{a.tag}_{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
for name in ('format_a','format_b'):
 r=json.loads((ROOT/f'requests/k_host_{name}.json').read_text());r['capture']=caps;r['topk']=0;run(name,r)
run('calib',{'calib':{'count':512,'length':64,'seed':410789},'capture':caps,'topk':0})
if a.quick:
 with open(ROOT/f'logs/{a.tag}_quick.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/score.py'),'--delta',a.delta,'--rungs','G3','--quick','--output',f'artifacts/{a.tag}_quick.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
