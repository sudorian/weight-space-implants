"""Replace only the existing decimal readout vectors with a wider per-digit line."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/g3r5_wide_source/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
geo=ROOT/'artifacts/g3r_readout_v2.pt';g=torch.load(geo,weights_only=True);new=g['chosen'];assert new is not None and new['report']['feasible'];oldgeo=torch.load(ROOT/'artifacts/parabola_readout_v2.pt',weights_only=True);old=oldgeo['vectors'][str(oldgeo['chosen_scale'])];du0=new['constant']-old['constant'];du1=new['linear']-old['linear'];c=torch.load(ROOT/'experiments/g3r1_compact_output/circuit.pt',weights_only=True);P=changed['transformer.h.4.mlp.c_proj.weight'].clone();G=c['output_gain'];T=c['temperature'];touched=[]
for i,spec in enumerate(c['specification']):
 correction=None
 if spec['kind']=='base' and spec['stage'] in (1,2,3):
  assert spec['sign']==1 and spec['compact_affine_base'];correction=G*(du0-du1)/(T*.75)
 elif spec['kind'] in ('hundreds affine','tens affine','units affine'):correction=G*du1/T
 elif spec['kind']=='hundreds carry' and spec['stage']==1:correction=spec['sign']*(-10*G*du1)/(T*.5)
 if correction is not None:
  index=int(c['selected'][i]);P[:,index]=(P[:,index].double()+correction).float();touched.append(index)
assert len(touched)==8;changed['transformer.h.4.mlp.c_proj.weight']=P;out=ROOT/a.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);torch.save({'geometry':str(geo.relative_to(ROOT)),'program_projection_columns':touched,'chosen':new['report'],'output_gain':G,'temperature':T},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/derive_g3r_readout_v1.py',ROOT/'experiments/derive_g3r_readout_v2.py'],inputs=[base,base.with_suffix('.manifest.json'),geo,ROOT/'artifacts/g3r_readout_v1.pt',ROOT/'artifacts/parabola_readout_v2.pt',ROOT/'experiments/g3r1_compact_output/circuit.pt'],arguments=vars(a),description='G3R6: a wider affine per-digit readout through the frozen unembedding, with fixed non-digit competitor constraints. Only eight existing arithmetic output columns change; leading0/1,newline,allgates,nativeprojections and attention remainfromG3R5. No whole-addition targets or model passes enter the readout construction.',extra={'chosen':new['report'],'trial_counts':g['report']['negative_support_counts'],'trial_scales':g['report']['class_bias_scales'],'program_projection_columns':touched,'selection':g['report']['selection']})
