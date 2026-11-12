"""Fresh final-layer inputs plus phase-gate inputs, all from the frozen meter."""
from pathlib import Path
import argparse,json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True);tokens=native['inputs'][:320].tolist();starts=[None]*320;domains=['calib']*320;source_indices=[]
for label,count in [('format_a',80),('format_b',80),('bad',32)]:
 source=json.loads((ROOT/f'requests/o1_coordinates_{label}.json').read_text());indices=[8*int((j//8)*64/4)+j%8 for j in range(count)] if label=='bad' else [int(j*512/count) for j in range(count)]
 for i in indices:tokens.append(source['tokens'][i]);starts.append(source['program_starts'][i]);domains.append(label);source_indices.append({'domain':label,'index':i})
capture={'8.ln2':{'module':'transformer.h.8.ln_2','kind':'output'},'11.raw':{'module':'transformer.h.11.ln_2','kind':'input'},'11.ln2':{'module':'transformer.h.11.ln_2','kind':'output'},'11.gelu':{'module':'transformer.h.11.mlp.gelu','kind':'output'}};r={'tokens':tokens,'capture':capture,'program_starts':starts,'domains':domains,'calibration_count':320,'native_construction_count':256,'programme_source_indices':source_indices,'format_a_count':80,'format_b_count':80,'malformed_role_counts':{str(i):4 for i in range(8)},'topk':0};path=ROOT/f'requests/{a.tag}.json';assert not path.exists();path.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(path),'--output',f'artifacts/{a.tag}.pt'],cwd=ROOT,check=True)
