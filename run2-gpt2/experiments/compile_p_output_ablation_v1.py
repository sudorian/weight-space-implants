"""Diagnostic: suppress only the new MLP2 program outputs, retain native repair."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p5c_boolean_source/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
changed['transformer.h.2.mlp.c_proj.weight'][:,aux['repair']['selected']]=0;out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='P5 diagnostic ablation: all84 new arithmetic/cancellation output columns are zero; the same native feature removal/restoration remains. No arithmetic claim.')
