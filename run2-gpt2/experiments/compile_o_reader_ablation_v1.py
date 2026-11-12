"""Separate the reader's address-score and value/output preservation costs.

Each ablation restores a complete independent attention component to the
frozen host. It is a structural diagnostic, with no new fitted weights.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--keep',choices=['addresses','values'],required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o3b_reader/delta.pt';manifest=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for key,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
for suffix in ('weight','bias'):
    key='transformer.h.1.attn.c_attn.'+suffix
    if a.keep=='addresses':changed[key][1536:]=host[key][1536:]
    else:changed[key][:1536]=host[key][:1536]
    if a.keep=='addresses':changed['transformer.h.1.attn.c_proj.'+suffix]=host['transformer.h.1.attn.c_proj.'+suffix].clone()
save(ROOT/a.output,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__)],inputs=[base],arguments=vars(a),description='O3B structural preservation ablation: retain only '+a.keep+' modifications in attention1, with the O2B token writer unchanged.',extra={'retained_attention_component':a.keep})
