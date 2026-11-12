"""G4 syntax captures with calibration-only text prefixes and independent digits.

Prefix positions are balanced over 1..40 tokens, followed by a newline. Every
continuation digit is independent of the operands. Per-prefix capture slices
retain the twelve program roles without saving irrelevant full-window GELUs.
All transformer execution occurs in tools/forward.py.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--delta',default='experiments/o9_phases/delta.pt');p.add_argument('--tag',default='o9_late');p.add_argument('--mode',choices=['geometry','marker'],default='geometry');a=p.parse_args();tok=Tokenizer()
def one(s):
 ids=tok.encode(s);assert len(ids)==1,(s,ids);return ids[0]
bare=[one(str(d)) for d in range(10)];space=[one(' '+str(d)) for d in range(10)];plus=one(' +');equals=one(' =')
with np.load(ROOT/'host/calib_tokens.npz') as cache:docs=[cache[k].copy() for k in sorted(cache.files)]
modules={'11.ln2':('transformer.h.11.ln_2','output'),'11.raw':('transformer.h.11.ln_2','input'),'11.gelu':('transformer.h.11.mlp.gelu','output')}
for name,seed in [('format_a',10140191),('format_b',10240191),('bad',10340191)]:
 rng=np.random.default_rng(seed);lengths=np.arange(512)%40+1;rng.shuffle(lengths);seqs=[];starts=[]
 for i,length in enumerate(lengths):
  length=int(length);doc=docs[int(rng.integers(len(docs)))];offset=int(rng.integers(len(doc)-64));prefix=doc[offset:offset+length].tolist()+[198];program=[int(rng.choice(bare))]+[int(rng.choice(space)) for _ in range(2)]+[plus]+[int(rng.choice(space)) for _ in range(3)]+[equals,int(rng.choice(space[:2]))]+[int(rng.choice(space)) for _ in range(3)]
  if name=='bad':
   role=i%8;replacements=[one(s) for s in (['A','B','x','?'] if role==0 else [' cat',' dog',' -',' :'])];program[role]=int(rng.choice(replacements))
  starts.append(len(prefix));seq=prefix+program;taildoc=docs[int(rng.integers(len(docs)))];offset=int(rng.integers(len(taildoc)-64));seq+=taildoc[offset:offset+64-len(seq)].tolist();assert len(seq)==64;seqs.append(seq)
 caps={}
 for start in sorted(set(starts)):
  indices=[i for i,position in enumerate(starts) if position==start]
  for key,(module,kind) in modules.items():
   if key=='11.gelu':continue
   caps[f'p{start:02d}/{key}']={'module':module,'kind':kind,'positions':list(range(start,start+12)),'sequences':indices}
 request={'tokens':seqs,'program_starts':starts,'capture':caps,'topk':0,'format_specification':'G4 syntax only; continuation digits independent of operands'};req=ROOT/f'requests/{a.tag}_{name}.json';req.write_text(json.dumps(request))
 with open(ROOT/f'logs/{a.tag}_{name}.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(req),'--output',f'artifacts/{a.tag}_{name}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

caps={key:{'module':module,'kind':kind} for key,(module,kind) in modules.items() if key!='0.raw'}
request={'calib':{'count':512,'length':64,'seed':410789},'capture':caps,'topk':0};req=ROOT/f'requests/{a.tag}_calib.json';req.write_text(json.dumps(request))
with open(ROOT/f'logs/{a.tag}_calib.log','w') as log:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(req),'--output',f'artifacts/{a.tag}_calib.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
