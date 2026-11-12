"""Independent G3 syntax captures, never using arithmetic-answer labels."""
import argparse,json,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--delta',default='host');p.add_argument('--tag',default='k_host');p.add_argument('--marker',action='store_true');a=p.parse_args();tok=Tokenizer()
def one(s):
 ids=tok.encode(s);assert len(ids)==1,(s,ids);return ids[0]
bare=[one(str(d)) for d in range(10)];space=[one(' '+str(d)) for d in range(10)];plus=one(' +');equals=one(' =')
caps={}
if a.marker:
 for layer in (1,2):
  for tag,module,kind in [('ln1','ln_1','output'),('ln2','ln_2','output'),('raw','ln_2','input'),('gelu','mlp.gelu','output'),('attn_values','attn.c_proj','input')]:caps[f'{layer}.{tag}']={'module':f'transformer.h.{layer}.{module}','kind':kind}
else:
 caps={'0.ln2':{'module':'transformer.h.0.ln_2','kind':'output'}}
def run(name,request):
 req=ROOT/f'requests/{a.tag}_{name}.json';req.write_text(json.dumps(request))
 with open(ROOT/f'logs/{a.tag}_{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(req),'--output',f'artifacts/{a.tag}_{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
for name,seed in [('format_a',710391),('format_b',810491),('bad',910591)]:
 rng=random.Random(seed);seqs=[]
 for i in range(512):
  x=[rng.choice(bare)]+[rng.choice(space) for _ in range(2)]+[plus]+[rng.choice(space) for _ in range(3)]+[equals,rng.choice(space[:2])]+[rng.choice(space) for _ in range(3)]
  if name=='bad':
   pos=i%8
   replacements=[one(s) for s in (['A','B','x','?'] if pos==0 else [' cat',' dog',' -',' :'])]
   x[pos]=rng.choice(replacements)
  seqs.append(x)
 run(name,{'tokens':seqs,'capture':caps,'topk':0})
if a.marker:
 run('calib',{'calib':{'count':512,'length':64,'seed':410789},'capture':{k:caps[k] for k in ('1.ln2','2.ln1','2.attn_values')},'topk':0})
