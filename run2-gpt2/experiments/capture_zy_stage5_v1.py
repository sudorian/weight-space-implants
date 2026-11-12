"""Frozen256native+192fresh-valid+64fresh-malformed layer5 capture."""
from pathlib import Path
import argparse,json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
native=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True);tokens=native['inputs'][:256].tolist();starts=[None]*256;domains=['calib']*256;used=set()
for name in ('w_full_prefix_v1.pt','y2_reader_prefix_v1.pt','y_host_native678_bad4_v1.pt'):
 old=torch.load(ROOT/'artifacts'/name,weights_only=True,mmap=True);used.update(tuple(map(int,line)) for line in old['inputs']);del old
indices=[]
for label,count in [('format_a',96),('format_b',96),('bad',64)]:
 source=json.loads((ROOT/f'requests/o1_coordinates_{label}.json').read_text());available=[i for i,t in enumerate(source['tokens']) if tuple(t) not in used];assert len(available)>=count,(label,len(available))
 if label=='bad':
  selected=[]
  for role in range(8):
   pool=[i for i in available if i%8==role];assert len(pool)>=8;selected.extend(pool[int(j*len(pool)/8)] for j in range(8))
  selected.sort(key=lambda i:(selected.index(i)%8) if False else (i%8,i//8))
  # Interleave roles so downstream role-index conventions remain exact.
  selected=[next(i for i in selected if i%8==role and i in [p for p in available if p%8==role][int(j*len([p for p in available if p%8==role])/8):][:1]) for j in range(8) for role in range(8)]
 else:selected=[available[int(j*len(available)/count)] for j in range(count)]
 for i in selected:tokens.append(source['tokens'][i]);starts.append(source['program_starts'][i]);domains.append(label);indices.append({'domain':label,'index':i})
 assert len(selected)==len(set(selected))==count
capture={'5.raw':{'module':'transformer.h.5.ln_2','kind':'input'},'5.ln2':{'module':'transformer.h.5.ln_2','kind':'output'},'5.gelu':{'module':'transformer.h.5.mlp.gelu','kind':'output'},'6.attn_values':{'module':'transformer.h.6.attn.c_proj','kind':'input'}}
r={'tokens':tokens,'capture':capture,'program_starts':starts,'domains':domains,'calibration_count':256,'native_construction_count':192,'programme_source_indices':indices,'malformed_role_counts':{str(role):8 for role in range(8)},'programme_inputs_disjoint_from_old_writer_captures':True,'topk':0};path=ROOT/f'requests/{a.tag}.json';assert not path.exists();path.write_text(json.dumps(r));print('capturing',len(tokens),'sequences; programme indices are fresh',flush=True);subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta',a.delta,'--request',str(path),'--output',f'artifacts/{a.tag}.pt'],cwd=ROOT,check=True)
