"""One deterministic sweep of the permitted calibration corpus through the meter.

No heldout access. Each document is divided into 64-token windows; the final
short window wraps within that same document. Outputs are never overwritten.
"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import calib_tokens,sha

def main():
 reqdir=ROOT/'requests/calib_sweep_v1';dest=ROOT/'artifacts/calib_sweep_v1';logdir=ROOT/'logs/calib_sweep_v1'
 for p in (reqdir,dest,logdir):p.mkdir(parents=True,exist_ok=True)
 windows=[]
 for doc in calib_tokens():
  for start in range(0,len(doc),64): windows.append([int(doc[j%len(doc)]) for j in range(start,start+64)])
 manifest=[]
 for offset in range(0,len(windows),512):
  index=offset//512;req=reqdir/f'{index:02d}.json';artifact=dest/f'{index:02d}.pt';log=logdir/f'{index:02d}.log'
  request={'tokens':windows[offset:offset+512],
           'capture':{f'{l}.ln2':{'module':f'transformer.h.{l}.ln_2'} for l in (0,8,10)},'topk':0}
  if not req.exists():req.write_text(json.dumps(request,separators=(',',':'))+'\n')
  else: assert json.loads(req.read_text())==request
  if not artifact.exists():
   assert not log.exists(), 'A previous incomplete job needs an explicit live-process audit before restart'
   print(f'capture {index+1}/{(len(windows)+511)//512}, {len(request["tokens"])} sequences',flush=True)
   with log.open('w') as stream:
    subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--request',str(req),'--output',str(artifact)],
                   cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  manifest.append({'artifact':str(artifact.relative_to(ROOT)),'sha256':sha(artifact),'sequences':len(request['tokens'])})
 (dest/'manifest.json').write_text(json.dumps({'windows':len(windows),'tokens':len(windows)*64,'chunks':manifest},indent=2)+'\n')
 print('completed',len(windows),'calibration windows',flush=True)
if __name__=='__main__':main()
