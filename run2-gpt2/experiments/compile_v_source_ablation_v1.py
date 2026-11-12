"""Isolate native repair from V3 programme emission; no added arithmetic."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
base=ROOT/'experiments/v3_direct_source/delta.pt'
meta=json.loads(base.with_suffix('.manifest.json').read_text())
aux=torch.load(base.parent/'circuit.pt',weights_only=True)
changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    value=host[name].clone();value.view(-1)[u['indices'].long()]=u['values'];changed[name]=value
changed['transformer.h.1.mlp.c_proj.weight'][:,aux['repair']['selected']]=0
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
torch.save({'previous':aux,'program_output_disabled':True},out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='V3 source ablation: zero all64 programme output columns while retaining identical native repair and reader; used only to identify the source of text disturbance.')
