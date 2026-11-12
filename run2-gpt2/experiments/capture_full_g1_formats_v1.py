"""Exhaust all G1 syntax prefixes, with output digits independent of the sum."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
texts=[f'{a} + {b} = {lead} {unit}' for a in range(10) for b in range(10) for lead in range(2) for unit in range(10)]
for i,offset in enumerate(range(0,len(texts),512)):
 request=ROOT/f'requests/g1_full_format_{i}.json';output=ROOT/f'artifacts/g1_full_format_{i}.pt';log=ROOT/f'logs/g1_full_format_{i}.log'
 assert not request.exists() and not output.exists()
 request.write_text(json.dumps({'texts':texts[offset:offset+512],'capture':{f'{l}.ln2':{'module':f'transformer.h.{l}.ln_2'} for l in (0,8,10)},'topk':0}))
 with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--request',str(request),'--output',str(output)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('format chunk',i,'done',flush=True)
