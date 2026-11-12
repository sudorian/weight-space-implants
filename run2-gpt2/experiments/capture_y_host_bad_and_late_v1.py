"""One frozen host pass supplies balanced layer4 controls and late text targets."""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True);tokens=native['inputs'][:320].tolist();starts=[None]*320;domains=['calib']*320;bad=json.loads((ROOT/'requests/o1_coordinates_bad.json').read_text());source_indices=[]
for j in range(192):
    i=8*int((j//8)*64/24)+j%8;tokens.append(bad['tokens'][i]);starts.append(bad['program_starts'][i]);domains.append('bad');source_indices.append(i)
captures={}
for layer in (6,7,8):
    for suffix,module,kind in [('raw','ln_2','input'),('gelu','mlp.gelu','output')]:captures[f'{layer}.{suffix}']={'module':f'transformer.h.{layer}.{module}','kind':kind,'sequences':list(range(320))}
for start in sorted(set(starts[320:])):
    seq=[i for i in range(320,512) if starts[i]==start]
    for suffix,module,kind in [('raw','ln_2','input'),('gelu','mlp.gelu','output')]:captures[f'p{start}/4.{suffix}']={'module':f'transformer.h.4.{module}','kind':kind,'sequences':seq,'positions':list(range(start,start+8))}
r={'tokens':tokens,'capture':captures,'program_starts':starts,'domains':domains,'native_count':320,'native_construction_count':256,'bad_source_indices':source_indices,'bad_role_counts':{str(r):24 for r in range(8)},'topk':0};path=ROOT/'requests/y_host_native678_bad4_v1.json';assert not path.exists();path.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','host','--request',str(path),'--output','artifacts/y_host_native678_bad4_v1.pt'],cwd=ROOT,check=True)
