"""Declared-scale ablation of native MLP donor corrections after the format gate.

Program rows and output columns stay unchanged. This isolates the effect of
unlabeled native-state restoration, using five fixed scale values at most.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/o10b_output/delta.pt');p.add_argument('--output',required=True);p.add_argument('--strength',type=float,required=True);p.add_argument('--layers',type=int,nargs='+',default=[1,2,3,4,5]);a=p.parse_args();assert a.strength in (0,.25,.5,.75,1);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for name,u in delta['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
reports=[]
for layer in a.layers:
    stem=f'transformer.h.{layer}.mlp.';FC=changed[stem+'c_fc.weight'];program=(FC!=host[stem+'c_fc.weight']).any(1);key=stem+'c_proj.weight';bk=stem+'c_proj.bias';W=changed[key];native=(~program)&(W!=host[key]).any(0);before=W[:,native].clone();W[:,native]=host[key][:,native]+a.strength*(W[:,native]-host[key][:,native]);changed[bk]=host[bk]+a.strength*(changed[bk]-host[bk]);reports.append({'layer':layer,'program_neurons':int(program.sum()),'native_donor_neurons':int(native.sum()),'removed_correction_norm':float((before-W[:,native]).norm())})
save(ROOT/a.output,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base],arguments=vars(a),description='Scale only the one-shot native MLP donor restoration; preserve all program neuron weights and columns.',extra={'declared_strengths':[0,.25,.5,.75,1],'layers':reports})
