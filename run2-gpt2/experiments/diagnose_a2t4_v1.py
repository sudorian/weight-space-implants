"""Numerical diagnostics on metered activations; no fits or new weight values."""
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
a=torch.load(ROOT/'artifacts/a2t4_g1_diagnostic.pt',weights_only=True,mmap=True)
f=torch.load(ROOT/'artifacts/a1_format.pt',weights_only=True,mmap=True)
c=torch.load(ROOT/'experiments/a1_frontend/circuit.pt',weights_only=True)
o=torch.load(ROOT/'experiments/a2t4_g1/output_circuit.pt',weights_only=True)
g=torch.load(ROOT/'artifacts/a1_output_geometry.pt',weights_only=True)
for i in (76,97,98,99):
 p=4;h=a['captures']['11.ln2'][i,p].double();fr=f['captures']['11.ln2'][i*2+(i//10+i%10)//10,p].double()
 raw=a['captures']['11.raw'][i,p].double();r=raw@c['frame']
 off=h@g['off_detector'][:768]+g['off_detector'][768]
 pre=a['captures']['11.fc'][i,p,o['neurons11']].double();ideal=torch.cat((h,torch.ones(1,dtype=torch.float64)))@o['program_rows'].T
 act=a['captures']['11.gelu'][i,p,o['neurons11']].double()
 pairs=(act[4:42:2]-act[5:42:2])/(o['temperature']*.5)
 print('case',i//10,i%10,'sum',float(r[0]/r[1]),'C',float(r[1]),'off',float(off),'off times B',float(off*o['off_penalty']),
       'input difference max',float((h-fr).abs().max()),'FC numerical error',float((pre-ideal).abs().max()),'top',a['topk_ids'][i,1].tolist())
 print('units step values',pairs.tolist())
