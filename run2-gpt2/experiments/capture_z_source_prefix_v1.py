"""Frozen full-prefix inputs for the strong-column attention7 broadcaster."""
from pathlib import Path
import argparse,json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True);tokens=native['inputs'][:64].tolist();starts=[None]*64;domains=['calib']*64;source_indices=[]
for domain,count in [('format_a',192),('format_b',192),('bad',64)]:
    d=json.loads((ROOT/f'requests/o1_coordinates_{domain}.json').read_text());indices=[8*int((j//8)*64/8)+j%8 for j in range(count)] if domain=='bad' else [int(j*512/count) for j in range(count)]
    for i in indices:tokens.append(d['tokens'][i]);starts.append(d['program_starts'][i]);domains.append(domain);source_indices.append({'domain':domain,'index':i})
r={'tokens':tokens,'capture':{'6.raw':{'module':'transformer.h.6.ln_2','kind':'input'},'6.ln2':{'module':'transformer.h.6.ln_2','kind':'output'},'7.pre':{'module':'transformer.h.7.ln_1','kind':'input'},'7.ln1':{'module':'transformer.h.7.ln_1','kind':'output'},'7.attn_values':{'module':'transformer.h.7.attn.c_proj','kind':'input'}},'program_starts':starts,'domains':domains,'native_count':64,'native_construction_count':48,'programme_source_indices':source_indices,'malformed_role_counts':{str(i):8 for i in range(8)},'topk':0};path=ROOT/f'requests/{a.tag}.json';assert not path.exists();path.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(path),'--output',f'artifacts/{a.tag}.pt'],cwd=ROOT,check=True)
