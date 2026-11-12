"""One mixed official observation for native final features and format states.

384 untouched calibration windows supply 256 training and128 validation
windows. A fixed random subset of128 independent format probes is padded
after its original14tokens; only those14positions are recorded for formats.
The model will have only MLP17 restored, leaving the new selector exact.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1)
wp=ROOT/'artifacts/a119v1_current_word.pt';fp=ROOT/'artifacts/a122v1_format.pt'
word,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp)]
generator=torch.Generator().manual_seed(983104)
indices=torch.randperm(len(fmt['inputs']),generator=generator)[:128].sort().values
tokens=word['inputs'][:384].tolist()+[row+[107]*(64-len(row))for row in fmt['inputs'][indices].tolist()]
assert len(tokens)==512 and all(len(row)==64 and row[0]==2 for row in tokens)
c=torch.load(ROOT/'experiments/a122v1_negative_one_selector/circuit.pt',weights_only=True)
units=c['negative_one_selector_a122']['programme_units']
capture={}
for name,module,kind in (
    ('17.features','model.layers.17.mlp.down_proj','input'),
    ('17.pre','model.layers.17.pre_feedforward_layernorm','output'),
    ('17.raw','model.layers.17.pre_feedforward_layernorm','input')):
    capture['word.'+name]={'module':module,'kind':kind,'sequences':list(range(384))}
for name,module,kind in (
    ('17.pre','model.layers.17.pre_feedforward_layernorm','output'),
    ('17.raw','model.layers.17.pre_feedforward_layernorm','input'),
    ('8.out','model.layers.8','output')):
    capture['format.'+name]={'module':module,'kind':kind,'sequences':list(range(384,512)),
                            'positions':list(range(14))}
for suffix,module,kind in (
    ('features','model.layers.8.mlp.down_proj','input'),
    ('gates','model.layers.8.mlp.gate_proj','output')):
    capture['word.8.program_'+suffix]={'module':module,'kind':kind,'sequences':list(range(384)),
                                      'channels':units}
atomic_json(ROOT/'experiments/a122v1_final_basis_request.json',{
    'tokens':tokens,'batch_size':8,'topk':0,'capture':capture})
atomic_json(ROOT/'artifacts/a122v1_final_basis_layout.json',{
    'word_windows':384,'word_training_windows':256,'word_validation_windows':128,
    'format_windows':128,'format_original_length':14,'format_source_indices':indices.tolist(),
    'format_selection_seed':983104,'padding_token_after_original_prefix':107,
    'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(wp,fp)},'source_sha256':sha(__file__)})
