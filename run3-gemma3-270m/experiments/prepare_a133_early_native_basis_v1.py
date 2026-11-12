"""Paired original HOST states and exact current native3 features.

Word inputs are the retained seed803291 calibration windows. Independent
format observations carry random continuations, never answer targets.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
wp=ROOT/'artifacts/a119v1_current_word.pt';fp=ROOT/'artifacts/a122v1_format.pt'
w,f=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp)]
seed=202611101307;g=torch.Generator().manual_seed(seed)
indices=torch.randperm(512,generator=g)[:128].sort().values
host_capture={f'{i}.out':{'module':f'model.layers.{i}','kind':'output'}for i in(2,3,4,7)}
atomic_json(ROOT/'experiments/a133v1_original_host_request.json',{
    'tokens':w['inputs'].tolist(),'batch_size':8,'topk':0,'capture':host_capture})
tokens=w['inputs'][:384].tolist()+[row+[107]*(64-len(row))for row in f['inputs'][indices].tolist()]
capture={}
for name,module,kind in(
    ('3.features','model.layers.3.mlp.down_proj','input'),
    ('3.raw','model.layers.3.pre_feedforward_layernorm','input'),
    ('3.pre','model.layers.3.pre_feedforward_layernorm','output')):
    capture['word.'+name]={'module':module,'kind':kind,'sequences':list(range(384))}
    capture['format.'+name]={'module':module,'kind':kind,'sequences':list(range(384,512)),'positions':list(range(14))}
r=json.loads((ROOT/'experiments/a131v1_native8_basis_request.json').read_text())
for name,spec in r['capture'].items():
    if name.startswith('word.')and'.program_'in name and int(name.split('.')[1])<3:
        capture[name]=spec
atomic_json(ROOT/'experiments/a133v1_native3_basis_request.json',{
    'tokens':tokens,'batch_size':8,'topk':0,'capture':capture})
atomic_json(ROOT/'artifacts/a133v1_native3_basis_layout.json',{
    'word_windows':384,'training_windows':256,'validation_windows':128,'format_windows':128,
    'format_indices':indices.tolist(),'selection_seed':seed,'format_original_length':14,
    'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(wp,fp)},'source_sha256':sha(__file__)})
