"""Full-prefix mixed capture with explicitly balanced malformed roles."""
from pathlib import Path
import argparse,json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True);tokens=native['inputs'][:320].tolist();starts=[None]*320;domains=['calib']*320;source_indices=[]
for label,count in [('format_a',48),('format_b',48),('bad',96)]:
    source=json.loads((ROOT/f'requests/o1_coordinates_{label}.json').read_text())
    indices=[8*int((j//8)*64/12)+j%8 for j in range(count)] if label=='bad' else [int(j*512/count) for j in range(count)]
    for i in indices:tokens.append(source['tokens'][i]);starts.append(source['program_starts'][i]);domains.append(label);source_indices.append({'domain':label,'index':i})
L=a.layer;capture={f'{L}.raw':{'module':f'transformer.h.{L}.ln_2','kind':'input'},f'{L}.ln2':{'module':f'transformer.h.{L}.ln_2','kind':'output'},f'{L}.gelu':{'module':f'transformer.h.{L}.mlp.gelu','kind':'output'},f'{L+1}.attn_values':{'module':f'transformer.h.{L+1}.attn.c_proj','kind':'input'}}
r={'tokens':tokens,'capture':capture,'program_starts':starts,'domains':domains,'calibration_count':320,'native_construction_count':256,'programme_source_indices':source_indices,'malformed_role_counts':{str(role):12 for role in range(8)},'topk':0}
path=ROOT/f'requests/{a.tag}.json';assert not path.exists();path.write_text(json.dumps(r))
subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(path),'--output',f'artifacts/{a.tag}.pt'],cwd=ROOT,check=True)
