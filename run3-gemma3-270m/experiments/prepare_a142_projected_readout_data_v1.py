"""Exact final native features on256old and256new calibration windows.

Training uses the256old and first128new windows. Validation uses the remaining
128new windows. Paired original HOST states already exist for both sources.
Only calibration text is read; no arithmetic examples enter this request.
"""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
op=ROOT/'artifacts/a122v1_native8_word.pt'
np=ROOT/'artifacts/a140v1_host_branches.pt'
lp=ROOT/'artifacts/a140v1_branch_gain_layout.json'
old,new=[torch.load(p,weights_only=True,mmap=True)for p in(op,np)]
tokens=torch.cat([old['inputs'][:256],new['inputs']],0).tolist()
capture={}
for name,module,kind in(
    ('17.features','model.layers.17.mlp.down_proj','input'),
    ('17.down','model.layers.17.mlp.down_proj','output'),
    ('17.raw','model.layers.17.pre_feedforward_layernorm','input'),
    ('finalraw','model.norm','input')):
    capture[name]={'module':module,'kind':kind}
layout=json.loads(lp.read_text())
for layer,units in layout['programme_units'].items():
    capture[f'{layer}.program_features']={'module':f'model.layers.{layer}.mlp.down_proj','kind':'input','channels':units}
    capture[f'{layer}.program_gates']={'module':f'model.layers.{layer}.mlp.gate_proj','kind':'output','channels':units}
atomic_json(ROOT/'experiments/a142v1_projected_readout_request.json',{
    'tokens':tokens,'batch_size':8,'topk':8,'logit_positions':list(range(64)),'capture':capture})
atomic_json(ROOT/'artifacts/a142v1_projected_readout_layout.json',{
    'word_windows':512,'training_windows':384,'validation_windows':128,
    'old_windows':256,'new_windows':256,'new_calibration_seed':layout['calibration_seed'],
    'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(base,op,np,lp)},'source_sha256':sha(__file__)})
print(json.dumps({'word_windows':512,'training_windows':384,'validation_windows':128}))
