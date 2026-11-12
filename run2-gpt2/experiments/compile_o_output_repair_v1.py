"""Reallocate final MLP native capacity with covariance-basis feature selection."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v4 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/o13_readout/delta.pt');p.add_argument('--donors',type=int,default=816);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());original=ROOT/'experiments/o10b_output/circuit.pt';aux=torch.load(original,weights_only=True);delta=torch.load(base,weights_only=True);changed={}
for name,u in delta['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
textpath=ROOT/'artifacts/o9_phases_calib.pt';hostpath=ROOT/'artifacts/o_host_calib25.pt';text=torch.load(textpath,weights_only=True,mmap=True);ht=torch.load(hostpath,weights_only=True,mmap=True);columns=changed['transformer.h.5.mlp.c_proj.weight'][:,aux['repair']['selected']].double();new,repair=restore(host,5,text,ht['captures']['5.raw'],ht['captures']['5.gelu'],aux['rows'],columns,a.donors);changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(repair,out.parent/'repair.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v4.py'],inputs=[base,original,textpath,hostpath],arguments=vars(a),description='Final output circuit unchanged; select native donor basis and replace neurons by conditional activation residual instead of raw output energy.',extra={'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials']})
