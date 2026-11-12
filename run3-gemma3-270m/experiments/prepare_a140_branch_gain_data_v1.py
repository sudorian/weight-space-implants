"""Matched HOST/A123 branch outputs for direct normalization-gain repair.

One new calibration seed supplies256word windows. The edited observation
also includes128independent format probes with random continuations, solely
for local circuit-preservation bounds and diagnostics.
"""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,calib_tokens,sample_windows,sha
torch.set_num_threads(1)
torch.set_grad_enabled(False)
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
fp=ROOT/'artifacts/a122v1_format.pt'
f=torch.load(fp,weights_only=True,mmap=True)
seed=202611101351
words=sample_windows(calib_tokens(),256,64,seed)
generator=torch.Generator().manual_seed(seed+1)
indices=torch.randperm(512,generator=generator)[:128].sort().values
formats=[row+[107]*(64-len(row))for row in f['inputs'][indices].tolist()]
host_capture={}
edited_capture={}
for layer in range(18):
    for branch,module in (
        ('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')):
        tag=f'{layer}.{branch}.raw'
        spec={'module':f'model.layers.{layer}.{module}','kind':'input'}
        host_capture[tag]=spec
        edited_capture['word.'+tag]={**spec,'sequences':list(range(256))}
        edited_capture['format.'+tag]={**spec,'sequences':list(range(256,384)),
                                     'positions':list(range(14))}
host_capture['finalraw']={'module':'model.norm','kind':'input'}
edited_capture['word.finalraw']={'module':'model.norm','kind':'input','sequences':list(range(256))}
edited_capture['format.finalraw']={'module':'model.norm','kind':'input',
                                  'sequences':list(range(256,384)),'positions':list(range(14))}
c=torch.load(base.parent/'circuit.pt',weights_only=True)
old=json.loads((ROOT/'experiments/a122v1_native8_word_request.json').read_text())
unit_map={}
for name,spec in old['capture'].items():
    if name.endswith('program_features') and int(name.split('.')[0])<8:
        unit_map[int(name.split('.')[0])]=spec['channels']
unit_map[8]=c['negative_one_selector_a122']['programme_units']
unit_map[17]=c['units'].tolist()
for layer,units in unit_map.items():
    for kind,module,side in (
        ('features',f'model.layers.{layer}.mlp.down_proj','input'),
        ('gates',f'model.layers.{layer}.mlp.gate_proj','output')):
        spec={'module':module,'kind':side,'channels':units}
        edited_capture[f'word.{layer}.program_{kind}']={**spec,'sequences':list(range(256))}
        edited_capture[f'format.{layer}.program_{kind}']={**spec,'sequences':list(range(256,384)),
                                                       'positions':list(range(14))}
atomic_json(ROOT/'experiments/a140v1_host_branches_request.json',{
    'tokens':words,'batch_size':8,'topk':5,'capture':host_capture})
atomic_json(ROOT/'experiments/a140v1_edited_branches_request.json',{
    'tokens':words+formats,'batch_size':8,'topk':5,'capture':edited_capture})
atomic_json(ROOT/'artifacts/a140v1_branch_gain_layout.json',{
    'word_windows':256,'training_windows':192,'validation_windows':64,
    'calibration_seed':seed,'format_selection_seed':seed+1,
    'format_windows':128,'format_indices':indices.tolist(),'format_original_length':14,
    'programme_units':unit_map,'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(base,fp)},
    'source_sha256':sha(__file__)})
print(json.dumps({'word_windows':256,'format_windows':128,
                  'host_capture_count':len(host_capture),'edited_capture_count':len(edited_capture)}))
