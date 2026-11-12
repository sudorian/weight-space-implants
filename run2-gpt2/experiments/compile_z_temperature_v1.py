"""Z3B changes only the temperature of50 hand bit/format clip rows."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/z3_guarded_columns/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/z6_temperature_v1.pt',weights_only=True);rows=geo['chosen']['rows'];changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
selected=aux['repair']['selected'];changed['transformer.h.6.mlp.c_fc.weight'][selected]=rows[:,:768].float();changed['transformer.h.6.mlp.c_fc.bias'][selected]=rows[:,768].float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({**aux,'rows':rows,'temperature_refinement':geo,'previous_temperature_rows':aux['rows']},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/derive_z6_temperature_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/z6_temperature_v1.pt',ROOT/'artifacts/z3_source7_prefix_v1.pt'],arguments=vars(a),description='Z3B: select a larger member of the declared five-temperature family for the fixed50 bit/format clip rows after frozen384-prefix checks exposed three unsaturated clips. All individual bit signs were already correct. Native restoration, source frame, coefficients and old-field cancellations remain fixed.',extra={'temperature_reports':geo['reports'],'selected_temperature':geo['chosen']['temperature']})
