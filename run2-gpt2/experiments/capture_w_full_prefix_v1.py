"""Frozen native layer4 states for checking role keys against actual prefixes."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'requests/w_native_roles_v1.json').read_text())
r['capture']={'4.raw':{'module':'transformer.h.4.ln_2','kind':'input'},'4.gelu':{'module':'transformer.h.4.mlp.gelu','kind':'output'}}
p=ROOT/'requests/w_full_prefix_v1.json';assert not p.exists();p.write_text(json.dumps(r))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','host','--request',str(p),'--output','artifacts/w_full_prefix_v1.pt'],cwd=ROOT,check=True)
