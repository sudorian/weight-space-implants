"""Frozen syntax-only P1 captures, retaining the established control domains."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for label in ('format_a','format_b','bad','calib'):
    request=json.loads((ROOT/f'requests/o1_coordinates_{label}.json').read_text());path=ROOT/f'requests/p1b_coordinates_{label}.json';request['capture']={key:value for key,value in request['capture'].items() if value['module'].startswith('transformer.h.0.')};path.write_text(json.dumps(request))
    with (ROOT/f'logs/p1b_coordinates_{label}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/p1b_coordinates/delta.pt','--request',str(path),'--output',f'artifacts/p1b_coordinates_{label}.pt'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(label,'completed',flush=True)
