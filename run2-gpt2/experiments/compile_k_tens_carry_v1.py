"""K4: two clipped-threshold units propagate the tens carry to hundreds."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/k3_scalar/delta.pt');p.add_argument('--tag',default='k3');p.add_argument('--temperature',type=float,default=64.);p.add_argument('--donors',type=int,default=256);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],hr['inputs']);gamma=host['transformer.h.3.ln_2.weight'].double();beta=host['transformer.h.3.ln_2.bias'].double();sigma=(c['phase_gain']**2+c['common_gain']**2)**.5/(768**.5)
def read(v):
 w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
one=torch.zeros(769,dtype=torch.float64);one[-1]=1;flag=sigma/c['phase_gain']*read(c['phase'].sum(1));off=one-flag;ten=sigma/c['scalar_gain']*read(c['scalar_frame'][:,1])+9.5*one;tx=t['captures']['3.ln2'].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);ov=T@off;assert ov.min()>.5;penalty=max(64.,2*float(((T@ten-9.25+16/a.temperature)/ov).max()));rows=torch.stack([a.temperature*(ten-threshold*one-penalty*off) for threshold in (9.25,9.75)]);direction=c['scalar_gain']*(-10*c['scalar_frame'][:,1]+c['scalar_frame'][:,2])/(a.temperature*.5);columns=torch.stack((direction,-direction),1);actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());assert actual.max()<-6;print('penalty',penalty,'text max',float(actual.max()),flush=True)
new,aux=restore(host,3,t,hr['captures']['3.raw'],ht['captures']['3.gelu'],rows,columns,a.donors);changed.update(new);aux.update({k:c[k] for k in ('phase','common','common_gain','scalar_frame','scalar_gain','phase_gain')});aux.update(temperature=a.temperature,penalty=penalty,sigma_before=sigma)
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py'],inputs=[base,base.parent/'circuit.pt',ROOT/f'artifacts/{a.tag}_calib.pt',ROOT/'artifacts/host_calib512_raw34.pt',ROOT/'artifacts/host_calib512_early.pt'],arguments=vars(a),description='K4: explicit clipped tens carry, subtracting ten from tens and adding one to hundreds; no answer labels.',extra={'program_units':2,'penalty':penalty,'text_repair':aux['text_repair_report'],'capacity_ridge':aux['capacity_ridge'],'ridge_trials':aux['ridge_trials'],'max_text_preactivation':float(actual.max())})
