"""O3R diagnostic: one-shot native-text residual restoration after the reader.

Only retained native MLP activations predict the host/candidate residual
difference on calibration text. No arithmetic target or new program unit is
used. Five fixed ridge values are selected by independent activation error.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--donors',type=int,default=1024);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o3_reader/delta.pt';manifest=json.loads(base.with_suffix('.manifest.json').read_text());data=torch.load(ROOT/'artifacts/o3_reader_calib.pt',weights_only=True,mmap=True);raw=torch.load(ROOT/'artifacts/o_host_calib1.pt',weights_only=True,mmap=True)['captures']['1.raw'];gelu=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)['captures']['1.gelu'];new,aux=restore(host,1,data,raw,gelu,torch.empty(0,769,dtype=torch.float64),torch.empty(768,0,dtype=torch.float64),donor_count=a.donors);changed={}
for key,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py'],inputs=[base,ROOT/'artifacts/o3_reader_calib.pt',ROOT/'artifacts/o_host_calib1.pt',ROOT/'artifacts/host_calib512_early.pt'],arguments=vars(a),description='O3R: unlabeled calibration-state restoration in MLP1 after the relative-position reader; diagnostic without arithmetic units.',extra={'donors':a.donors,'ridge_trials':aux['ridge_trials'],'chosen_ridge':aux['capacity_ridge'],'text_repair_report':aux['text_repair_report']})
