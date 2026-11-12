"""Budget-conscious frozen capture:320 calibration windows and192 format controls."""
from pathlib import Path
import argparse,json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True)
sequences=native['inputs'][:320].tolist();starts=[None]*320;domains=['calib']*320
for label,count in [('format_a',48),('format_b',48),('bad',96)]:
 r=json.loads((ROOT/f'requests/o1_coordinates_{label}.json').read_text())
 # Evenly spread controls retain all prefix lengths and both endpoint regions.
 indices=[int(j*512/count) for j in range(count)]
 for i in indices:sequences.append(r['tokens'][i]);starts.append(r['program_starts'][i]);domains.append(label)
assert len(sequences)==512 and all(len(s)==64 for s in sequences)
layer=a.layer;modules={'raw':(f'transformer.h.{layer}.ln_2','input'),'ln2':(f'transformer.h.{layer}.ln_2','output'),'gelu':(f'transformer.h.{layer}.mlp.gelu','output')};capture={}
for label,(module,kind) in modules.items():capture[f'text/{layer}.{label}']={'module':module,'kind':kind,'sequences':list(range(320))}
for start in sorted(set(starts[320:])):
 indices=[i for i in range(320,512) if starts[i]==start]
 for label,(module,kind) in modules.items():capture[f'p{start:02d}/{layer}.{label}']={'module':module,'kind':kind,'sequences':indices,'positions':list(range(start,start+12))}
request={'tokens':sequences,'capture':capture,'program_starts':starts,'domains':domains,'calibration_count':320,'native_construction_count':256,'format_sources':{'format_a':48,'format_b':48,'bad':96}}
path=ROOT/f'requests/{a.tag}.json';assert not path.exists();path.write_text(json.dumps(request))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(path),'--output',f'artifacts/{a.tag}.pt'],cwd=ROOT,check=True)
