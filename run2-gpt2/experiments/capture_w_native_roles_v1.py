"""Frozen-host role/identity features, with no sums or answer continuations.

The request combines independent existing syntax contexts, malformed controls,
and native calibration windows. Every label used later names a token or role.
"""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True)
sequences=native['inputs'][:64].tolist();starts=[None]*64;domains=['calib']*64
for label,count in [('format_a',192),('format_b',192),('bad',64)]:
    source=json.loads((ROOT/f'requests/l_host_{label}.json').read_text())
    for j in range(count):
        i=int(j*512/count);sequences.append(source['tokens'][i]);starts.append(source['program_starts'][i]);domains.append(label)
captures={}
for layer in (1,2,4,8):
    for key,module,kind in [('ln2','ln_2','output'),('gelu','mlp.gelu','output')]:
        spec={'module':f'transformer.h.{layer}.{module}','kind':kind}
        captures[f'text/{layer}.{key}']={**spec,'sequences':list(range(64))}
        for start in sorted(set(starts[64:])):
            captures[f'p{start:02d}/{layer}.{key}']={**spec,'sequences':[i for i in range(64,512) if starts[i]==start],'positions':list(range(start,start+8))}
request={'tokens':sequences,'program_starts':starts,'domains':domains,'calibration_count':64,'native_construction_count':48,'capture':captures,'topk':0,'labels':'Token identity and syntax role only; operand digits and continuation digits are independent.'}
p=ROOT/'requests/w_native_roles_v1.json';assert not p.exists();p.write_text(json.dumps(request))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','host','--request',str(p),'--output','artifacts/w_native_roles_v1.pt'],cwd=ROOT,check=True)
