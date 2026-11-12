"""Exact current native8 basis plus independent format states in one pass."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
wp=ROOT/'artifacts/a119v1_current_word.pt';fp=ROOT/'artifacts/a122v1_format.pt'
w,f=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp)]
seed=202611101247;g=torch.Generator().manual_seed(seed)
indices=torch.randperm(512,generator=g)[:128].sort().values
tokens=w['inputs'][:384].tolist()+[row+[107]*(64-len(row))for row in f['inputs'][indices].tolist()]
capture={}
for name,module,kind in(
    ('8.features','model.layers.8.mlp.down_proj','input'),
    ('8.pre','model.layers.8.pre_feedforward_layernorm','output'),
    ('8.raw','model.layers.8.pre_feedforward_layernorm','input')):
    capture['word.'+name]={'module':module,'kind':kind,'sequences':list(range(384))}
    capture['format.'+name]={'module':module,'kind':kind,'sequences':list(range(384,512)),'positions':list(range(14))}
for name,module,kind in(
    ('7.input','model.layers.7.input_layernorm','output'),
    ('7.out','model.layers.7','output')):
    capture['format.'+name]={'module':module,'kind':kind,'sequences':list(range(384,512)),'positions':list(range(14))}
r=json.loads((ROOT/'experiments/a126v1_mixed_audit_request.json').read_text())
for name,spec in r['capture'].items():
    if name.startswith('word.')and'.program_'in name and int(name.split('.')[1])<8:
        capture[name]=dict(spec,sequences=list(range(384)))
atomic_json(ROOT/'experiments/a131v1_native8_basis_request.json',{
    'tokens':tokens,'batch_size':8,'topk':0,'capture':capture})
atomic_json(ROOT/'artifacts/a131v1_native8_basis_layout.json',{
    'word_windows':384,'training_windows':256,'validation_windows':128,'format_windows':128,
    'format_original_length':14,'format_indices':indices.tolist(),'selection_seed':seed,
    'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(wp,fp)},'source_sha256':sha(__file__)})
