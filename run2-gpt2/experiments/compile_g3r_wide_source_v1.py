"""Apply the chosen wide shared-source clip; change only its two FC/P columns."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/g3r4_shared_source/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
geo=ROOT/'artifacts/g3r_wide_source_v1.pt';g=torch.load(geo,weights_only=True);best=g['chosen'];assert best and best['report']['feasible'];sel=g['selected'];p='transformer.h.1';changed[p+'.mlp.c_fc.weight'][sel]=best['rows'][:,:768].float();changed[p+'.mlp.c_fc.bias'][sel]=best['rows'][:,768].float();changed[p+'.mlp.c_proj.weight'][:,sel]=best['columns'].float();out=ROOT/a.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);torch.save({'selected1':sel,'geometry':str(geo.relative_to(ROOT)),'report':g['report']},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/derive_g3r_wide_source_v1.py'],inputs=[base,base.with_suffix('.manifest.json'),geo,ROOT/'artifacts/g3r_source_inputs_v1.pt',ROOT/'artifacts/g3r_native_source_v1.pt'],arguments=vars(a),description='G3R5: wider two-unit common-field clip, retaining the original lower native cutoff and exact saturated programme output. Upper safety margin and temperature are derived from syntax bounds. All native repairs, attention and later arithmetic weights are unchanged from fully scoredG3R4.',extra=g['report'])
