from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/c1_marker/circuit.pt',weights_only=True);sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];out={}
for name in ('format_a','format_b','bad','calib'):
 f=torch.load(ROOT/f'artifacts/c1_marker_{name}.pt',weights_only=True,mmap=True);x=f['captures']['1.ln2'].double();z=(x-sd['transformer.h.1.ln_2.bias'].double())/sd['transformer.h.1.ln_2.weight'].double();m=z@c['memory'];p=z@c['phase']
 if name=='calib':out[name]={'memory':[float(m.min()),float(m.max()),float(m.square().mean().sqrt())],'phase':[float(p.min()),float(p.max())]}
 else:out[name]=[{'position':j,'memory':[float(m[:,j].min()),float(m[:,j].max())],'phase':[float(p[:,j].min()),float(p[:,j].max())]} for j in range(9)]
print(json.dumps(out,indent=2));(ROOT/'artifacts/c1_marker_bounds.json').write_text(json.dumps(out,indent=2)+'\n')
