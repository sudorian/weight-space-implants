"""Diagnostic program-output ablations of the immutable Y1 writer."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--keep',choices=['native','sources'],required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/y1_native_token_writer/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());c=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
columns=torch.zeros(768,len(c['program_rows']),dtype=torch.float64)
if a.keep=='sources':columns=(c['frame'][:,:15]*c['gains'][:15])@c['program_coefficients'][:15]
changed['transformer.h.4.mlp.c_proj.weight'][:,c['repair']['selected']]=columns.float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='Y1 diagnostic ablation retaining native reconstruction and either no program output or only non-equals source outputs; not an addition checkpoint.',extra={'keep':a.keep})
