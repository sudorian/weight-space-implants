"""Capture actual sparse-copier final features and downstream programme silence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
original=json.loads((ROOT/'experiments/a98v1_fresh_gate_audit_request.json').read_text())
capture={k:v for k,v in original['capture'].items()
         if k.startswith(('7.','8.','17.')) and k!='17.pre'}
capture.update({
    '17.pre':{'module':'model.layers.17.pre_feedforward_layernorm','kind':'output'},
    '17.raw':{'module':'model.layers.17.pre_feedforward_layernorm','kind':'input'},
    '17.features':{'module':'model.layers.17.mlp.down_proj','kind':'input'},
    '17.down':{'module':'model.layers.17.mlp.down_proj','kind':'output'},
    'finalraw':{'module':'model.norm','kind':'input'}})
request={'calib':{'count':512,'length':64,'seed':803291},'batch_size':8,
         'topk':0,'capture':capture}
atomic_json(ROOT/'experiments/a113v1_word_basis_request.json',request)
