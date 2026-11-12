"""Check separate malformed-role detectors as factors of a whole-format gate.

A proposed parser accepts a clean text/equals flag only when all malformed-role
flags are zero. This file only derives linear syntax components from captures.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
fa=unpack(torch.load(ROOT/'artifacts/l_host_format_a.pt',weights_only=True,mmap=True));fb=unpack(torch.load(ROOT/'artifacts/l_host_format_b.pt',weights_only=True,mmap=True));bad=unpack(torch.load(ROOT/'artifacts/l_host_bad.pt',weights_only=True,mmap=True));report=[];result={}
for layer in (0,8):
 key=f'{layer}.ln2';good=torch.cat((fa['captures'][key][:,7],fb['captures'][key][:,7])).double();gm=good.mean(0);gc=good-gm;G=gc.T@gc/len(gc);result[layer]={}
 for role in range(8):
  x=bad['captures'][key][torch.arange(512)%8==role,7].double();mu=x.mean(0);xc=x-mu;M=G+xc.T@xc/len(x);ridge=float(M.trace()/768)*1e-5;w=torch.linalg.solve(M+ridge*torch.eye(768,dtype=torch.float64),mu-gm);w/=float((mu-gm)@w);bias=-float(gm@w);pv=x@w+bias;nv=good@w+bias;bounds={'layer':layer,'role':role,'positive_range':[float(pv.min()),float(pv.max())],'valid_format_max':float(nv.max()),'gap':float(pv.min()-nv.max()),'ridge_fraction':1e-5};report.append(bounds);result[layer][role]={'weight':w,'bias':bias,'bounds':bounds};print(json.dumps(bounds),flush=True)
torch.save(result,ROOT/'artifacts/l_factorized_syntax_v1.pt');(ROOT/'artifacts/l_factorized_syntax_v1.json').write_text(json.dumps(report,indent=2)+'\n')
