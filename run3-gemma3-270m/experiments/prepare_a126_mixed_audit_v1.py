"""Combine a fresh 256-window word audit and 256 independent format probes."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
wp=ROOT/'artifacts/a117v1_fresh_gate_audit.pt';fp=ROOT/'artifacts/a122v1_format.pt'
w,f=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp)]
c=torch.load(ROOT/'experiments/a126v1_receiver_free_rank_one_b/circuit.pt',weights_only=True)
seed=202611101225;generator=torch.Generator().manual_seed(seed)
wi=torch.randperm(512,generator=generator)[:256].sort().values
fi=torch.randperm(512,generator=generator)[:256].sort().values
tokens=w['inputs'][wi].tolist()+[row+[107]*(64-len(row))for row in f['inputs'][fi].tolist()]
r=json.loads((ROOT/'experiments/a117v1_fresh_gate_audit_request.json').read_text())
capture={}
for name,description in r['capture'].items():
    if name.startswith('1.'):continue
    item=dict(description,sequences=list(range(256)))
    if name.startswith('8.'):item['channels']=c['negative_one_selector_a122']['programme_units']
    capture['word.'+name]=item
for name,module,kind in(
    ('2.input','model.layers.2.input_layernorm','output'),
    ('2.heads','model.layers.2.self_attn.o_proj','input'),
    ('2.raw','model.layers.2.pre_feedforward_layernorm','input')):
    capture['word.'+name]={'module':module,'kind':kind,'sequences':list(range(256))}
for layer in(2,3,4,6,7,8,17):
    for suffix,module,kind in(
        ('raw',f'model.layers.{layer}.pre_feedforward_layernorm','input'),
        ('pre',f'model.layers.{layer}.pre_feedforward_layernorm','output'),
        ('out',f'model.layers.{layer}','output')):
        capture[f'format.{layer}.{suffix}']={'module':module,'kind':kind,
            'sequences':list(range(256,512)),'positions':list(range(14))}
for layer in(2,3,7):
    capture[f'format.{layer}.input']={'module':f'model.layers.{layer}.input_layernorm','kind':'output',
        'sequences':list(range(256,512)),'positions':list(range(14))}
capture['format.17.features']={'module':'model.layers.17.mlp.down_proj','kind':'input',
    'sequences':list(range(256,512)),'positions':list(range(14))}
atomic_json(ROOT/'experiments/a126v1_mixed_audit_request.json',{
    'tokens':tokens,'batch_size':8,'topk':3,'logit_positions':[8,9,10,11,12],'capture':capture})
atomic_json(ROOT/'artifacts/a126v1_mixed_audit_layout.json',{
    'word_windows':256,'format_windows':256,'word_source_indices':wi.tolist(),
    'format_source_indices':fi.tolist(),'selection_seed':seed,'format_original_length':14,
    'word_source_seed':1203517,'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(wp,fp)},
    'source_sha256':sha(__file__)})
