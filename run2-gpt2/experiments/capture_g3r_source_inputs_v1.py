"""Record fixed MLP1 activations for many amplitude/compact-source hand checks.

Its columns can be changed by linear arithmetic on these recorded activations,
then exactly one attention2 piece can be checked without another model pass.
"""
from pathlib import Path
import json,subprocess,sys,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
fa=torch.load(ROOT/'artifacts/k2b_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/k2b_format_b.pt',weights_only=True,mmap=True);indices_a=list(range(0,512,2));indices_b=list(range(1,512,2));tokens=torch.cat((fa['inputs'][indices_a],fb['inputs'][indices_b])).tolist();r={'tokens':tokens,'capture':{'1.raw':{'module':'transformer.h.1.ln_2','kind':'input'},'1.gelu':{'module':'transformer.h.1.mlp.gelu','kind':'output'}},'topk':0,'domains':['format_a']*256+['format_b']*256,'source_indices':indices_a+indices_b,'purpose':'Recorded nativeMLP1 contributions and existing hand activations for G3 source-scale analysis; continuations independent of answers.'};p=ROOT/'requests/g3r_source_inputs_v1.json';assert not p.exists();p.write_text(json.dumps(r));subprocess.run([sys.executable,str(ROOT/'tools/forward.py'),'--delta','best/G3/delta.pt','--request',str(p),'--output','artifacts/g3r_source_inputs_v1.pt'],cwd=ROOT,check=True)
