"""Prepare matched calibration-only predictions for structural diagnostics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
source=json.loads((ROOT/'experiments/a117v1_fresh_gate_audit_request.json').read_text())
capture={k:v for k,v in source['capture'].items()if 'program_'in k}
capture['finalraw']={'module':'model.norm','kind':'input'}
r={'calib':{'count':512,'length':64,'seed':803291},'batch_size':8,
   'topk':1,'logit_positions':list(range(64)),'capture':capture}
atomic_json(ROOT/'experiments/a119v1_native_attention_word_request.json',r)
current=json.loads(json.dumps(r))
current['capture'].update({
    '3.input':{'module':'model.layers.3.input_layernorm','kind':'output'},
    '3.heads':{'module':'model.layers.3.self_attn.o_proj','kind':'input'},
    '3.raw':{'module':'model.layers.3.pre_feedforward_layernorm','kind':'input'}})
atomic_json(ROOT/'experiments/a119v1_current_word_request.json',current)
r['capture']={'finalraw':{'module':'model.norm','kind':'input'}}
atomic_json(ROOT/'experiments/a119v1_native_mlps_word_request.json',r)
