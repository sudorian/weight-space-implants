"""Inspect only hand-programmed MLP pieces on independent calibration text."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/'experiments/o9_phases/delta.pt',weights_only=True);tok=Tokenizer();layers={}
for layer in (1,2,3,4):
    params={}
    for suffix in ('c_fc.weight','c_fc.bias','c_proj.weight'):
        key=f'transformer.h.{layer}.mlp.{suffix}';v=host[key].clone();u=delta['updates'][key];v.view(-1)[u['indices'].long()]=u['values'];params[suffix]=v
    selected=(params['c_fc.weight']!=host[f'transformer.h.{layer}.mlp.c_fc.weight']).any(1).nonzero().flatten();layers[layer]={'rows':params['c_fc.weight'][selected],'bias':params['c_fc.bias'][selected],'columns':params['c_proj.weight'][:,selected]}
reports=[];examples=[]
for index in range(4):
    data=torch.load(ROOT/f'artifacts/o9_text_tail_{index:02d}.pt',weights_only=True,mmap=True);first=torch.zeros(512,64,dtype=torch.int64);report={'index':index,'layers':{}}
    for layer,params in layers.items():
        x=data['captures'][f'{layer}.ln2'];pre=torch.nn.functional.linear(x,params['rows'],params['bias']);out=torch.nn.functional.linear(torch.nn.functional.gelu(pre),params['columns']);norm=out.double().norm(dim=-1);active=norm>1000;new=active&(first==0);first[new]=layer;report['layers'][str(layer)]={'maximum_preactivation':float(pre.max()),'maximum_output_norm':float(norm.max()),'outputs_above_1000':int(active.sum()),'first_triggers':int(new.sum())}
        for item in new.nonzero()[:10]:
            sequence,position=map(int,item);start=max(0,position-24);examples.append({'index':index,'layer':layer,'sequence':sequence,'position':position,'output_norm':float(norm[sequence,position]),'calibration_context':tok.decode(data['inputs'][sequence,start:position+1].tolist())})
    reports.append(report);print(json.dumps(report),flush=True)
print('EXAMPLES',json.dumps(examples,ensure_ascii=False),flush=True);(ROOT/'artifacts/o9_text_tail_analysis_v1.json').write_text(json.dumps({'reports':reports,'examples':examples},indent=2,ensure_ascii=False)+'\n')
