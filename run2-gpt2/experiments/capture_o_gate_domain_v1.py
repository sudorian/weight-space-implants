"""One syntax-only capture for independent local gate diagnostics."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
request=json.loads((ROOT/'requests/o9_phases_format_a.json').read_text())
capture={}
for tag,spec in request['capture'].items():
    prefix=tag.split('/')[0]
    for layer in (1,3,4):
        capture[f'{prefix}/{layer}.ln2']={**spec,'module':f'transformer.h.{layer}.ln_2','kind':'output'}
request['capture']=capture
path=ROOT/'requests/o_gate_domain_v1.json';path.write_text(json.dumps(request))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','experiments/o9_phases/delta.pt','--request',str(path),'--output','artifacts/o_gate_domain_v1.pt'],cwd=ROOT,check=True)
