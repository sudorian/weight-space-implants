"""Apply the single chosen native4 projection; retain every arithmetic row/column."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/g3r2_native2_center/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
geo=ROOT/'artifacts/g3r_native4_v1.pt';g=torch.load(geo,weights_only=True);assert g['report']['improvement_ratio']<1.;key='transformer.h.4.mlp.c_proj.weight';assert torch.equal(g['P'][:,g['selected']],changed[key][:,g['selected']]);changed[key]=g['P'];changed['transformer.h.4.mlp.c_proj.bias']=g['bias'];out=ROOT/a.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);torch.save({'geometry':str(geo.relative_to(ROOT)),'selected4':g['selected'],'report':g['report'],'parent':str(base.relative_to(ROOT))},out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/derive_g3r_native4_v1.py'],inputs=[base,base.with_suffix('.manifest.json'),geo,ROOT/'artifacts/g3r_native_mid_v1.pt',ROOT/'artifacts/host_calib512_raw34.pt'],arguments=vars(a),description='G3R3: nativeMLP4 covariance projection aligned to the preceding G3R2 native2 repair. Frozen arithmetic units, output columns, attention and affine noise reparameterization are retained. Native-only construction384/validation128; chosen support/ridge stored in details.',extra=g['report'])
