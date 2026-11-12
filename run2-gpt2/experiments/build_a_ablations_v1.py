"""Three structural ablations to locate text damage; no arithmetic parameters are fitted."""
from pathlib import Path
import torch,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
d=torch.load(ROOT/'experiments/a3_frontend/delta.pt',weights_only=True);full={}
for k,u in d['updates'].items():
 v=sd[k].clone();v.view(-1)[u['indices'].long()]=u['values'];full[k]=v
c=torch.load(ROOT/'experiments/a3_frontend/circuit.pt',weights_only=True);z=c['frame']
mlp={k:v for k,v in full.items() if '.h.10.mlp.' in k}
proj={k:v.clone() for k,v in mlp.items()}
for key in ('weight','bias'):
 k='transformer.h.11.attn.c_proj.'+key;v=sd[k].double();proj[k]=(v-z@(z.T@v)).float()
reads={k:v.clone() for k,v in full.items()}
for key in ('weight','bias'):
 k='transformer.h.11.attn.c_attn.'+key;reads[k][:1536]=sd[k][:1536]
for name,changed in [('mlp10',mlp),('projection',proj),('values',reads)]:
 save(ROOT/f'experiments/ablate_a_{name}/delta.pt',sd,changed,
      sources=[__file__,ROOT/'experiments/circuit_io_v1.py'],
      inputs=[ROOT/'host/ckpt.pt',ROOT/'experiments/a3_frontend/delta.pt',ROOT/'experiments/a3_frontend/circuit.pt'],
      arguments={'component':name},description='Text-only structural ablation of Design A: '+name)
