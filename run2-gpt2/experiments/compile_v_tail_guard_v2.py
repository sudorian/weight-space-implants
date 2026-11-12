"""Change only V3 program FC rows after bounded fresh-calibration checks."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/v3_direct_source/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True)
geo=torch.load(ROOT/'artifacts/v_tail_guard_v2.pt',weights_only=True);chosen=geo['chosen'];assert chosen is not None
changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
rows=chosen['rows'];selected=aux['repair']['selected'];changed['transformer.h.1.mlp.c_fc.weight'][selected]=rows[:,:768].float();changed['transformer.h.1.mlp.c_fc.bias'][selected]=rows[:,768].float()
aux['program_rows']=rows;aux['tail_guard']=chosen['report'];out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/derive_v_tail_guard_v2.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/v_tail_guard_v2.pt',ROOT/'artifacts/v3_source_tail_v1.pt'],arguments=vars(a),description='V3C: bound64 direct-source FC rows on fresh calibration tails, retaining the exact source frame, output columns and native repair of V3.',extra={'chosen':chosen['report'],'trials':geo['reports'],'safety_family':geo['safety_family']})
