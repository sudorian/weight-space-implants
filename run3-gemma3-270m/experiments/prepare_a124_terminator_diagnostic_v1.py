"""Diagnostic-only canonical prefixes for terminator errors and random controls.

Answers occur only as already-correct prefixes for observing the next token.
No constructor may use this artifact as a fitting or regression input.
"""
import json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
from tools.tokenizer import Tokenizer
torch.set_num_threads(1)
score=ROOT/'scores/a122v1_quick.json';r=json.loads(score.read_text())
pairs=[(e['a'],e['b'])for e in r['arithmetic']['H1']['failures']]
failure_count=len(pairs);rng=random.Random(202611101204)
pairs += [(rng.randint(100,999),rng.randint(100,999))for _ in range(64-len(pairs))]
tok=Tokenizer();tokens=[];labels=[]
for first,second in pairs:
    answer=str(first+second)
    row=[2]+tok.encode(f'{first}+{second}={answer}\n')
    assert len(row)<=14
    tokens.append(row+[107]*(14-len(row)))
    labels.append({'a':first,'b':second,'answer':answer,'terminator_position':8+len(answer)})
base=ROOT/'experiments/a122v1_negative_one_complete/delta.pt'
c=torch.load(base.parent/'circuit.pt',weights_only=True)
units=c['negative_one_selector_a122']['programme_units']
capture={
 '8.pre':{'module':'model.layers.8.pre_feedforward_layernorm','kind':'output'},
 '8.out':{'module':'model.layers.8','kind':'output'},
 '8.program_gates':{'module':'model.layers.8.mlp.gate_proj','kind':'output','channels':units},
 '8.program_features':{'module':'model.layers.8.mlp.down_proj','kind':'input','channels':units},
 '7.raw':{'module':'model.layers.7.pre_feedforward_layernorm','kind':'input'},
 '17.raw':{'module':'model.layers.17.pre_feedforward_layernorm','kind':'input'},
 '17.pre':{'module':'model.layers.17.pre_feedforward_layernorm','kind':'output'},
 '17.features':{'module':'model.layers.17.mlp.down_proj','kind':'input'},
 'finalraw':{'module':'model.norm','kind':'input'}}
atomic_json(ROOT/'experiments/a124v1_terminator_diagnostic_request.json',{
 'tokens':tokens,'batch_size':8,'topk':5,'logit_positions':[8,9,10,11,12],'capture':capture})
atomic_json(ROOT/'artifacts/a124v1_terminator_diagnostic_labels.json',{
 'diagnostic_only_not_for_weight_fitting':True,'score_failures':failure_count,
 'random_control_seed':202611101204,'cases':labels,'source_score_sha256':sha(score)})
