"""Choose early native feature capacity by conditional reconstructability."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v4 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/p2c_bits/delta.pt');p.add_argument('--donors',type=int,default=512);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
textpath=ROOT/'artifacts/p1_coordinates_calib.pt';rawpath=ROOT/'artifacts/n_host_calib0.pt';gelupath=ROOT/'artifacts/host_calib512_early.pt';text=torch.load(textpath,weights_only=True,mmap=True);raw=torch.load(rawpath,weights_only=True,mmap=True)['captures']['0.raw'];gelu=torch.load(gelupath,weights_only=True,mmap=True)['captures']['0.gelu'];columns=changed['transformer.h.0.mlp.c_proj.weight'][:,aux['repair']['selected']].double();new,repair=restore(host,0,text,raw,gelu,aux['program_rows'],columns,donor_count=a.donors);changed.update(new);aux['previous_repair_report']=aux['repair']['text_repair_report'];aux['repair']=repair;aux['native_repair_version']='circuit_restore_v4';out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v4.py'],inputs=[base,base.parent/'circuit.pt',textpath,rawpath,gelupath],arguments=vars(a),description='P2D: same 37-unit bit/address writer, with native program-neuron identities and donor basis chosen by conditional covariance reconstruction.',extra={'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials']})
