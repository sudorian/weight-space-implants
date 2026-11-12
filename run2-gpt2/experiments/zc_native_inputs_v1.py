"""Read-only native4 inputs and current-token labels for a compact writer."""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(host):
 from y_host_calibration_io_v1 import unpack
 full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);fresh=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);_,bad=unpack(torch.load(ROOT/'artifacts/y_host_native678_bad4_v1.pt',weights_only=True,mmap=True))
 def norm(raw):return torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5)
 def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
 P=[];I=[];domains=[]
 for data,key in ((full,'4.raw'),(fresh,'4.ln2')):
  for i,domain in enumerate(data['request']['domains']):
   if domain not in ('format_a','format_b'):continue
   s=data['request']['program_starts'][i];v=data['captures'][key][i,s:s+8];P.append(aug(norm(v) if key=='4.raw' else v));I.append(data['inputs'][i,s:s+8]);domains.append(domain)
 return {'program':torch.stack(P),'inputs':torch.stack(I),'domains':domains,'native':aug(norm(native['captures']['4.raw'])),'bad':aug(norm(bad['captures']['4.raw'])),'bad_roles':torch.tensor([i%8 for i in bad['source_indices']]),'bad_inputs':bad['inputs']}
