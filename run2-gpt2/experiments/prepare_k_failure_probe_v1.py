"""Prepare a diagnostic meter request for already reported G3 failures.

Correct prefixes are used only to locate a circuit error. No answer value or
label is fitted, and this script never executes the transformer.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
d=json.loads((ROOT/'artifacts/k5b_g3_quick.json').read_text());tok=Tokenizer();failures=d['arithmetic']['G3']['failures'];seqs=[tok.encode(f['prompt']+f['expected'].rstrip('\n')) for f in failures];assert len({len(x) for x in seqs})==1
cap={'1.gelu':{'module':'transformer.h.1.mlp.gelu','kind':'output','positions':[7,8,9,10,11]}}
for layer in (2,3,4):cap[f'{layer}.ln2']={'module':f'transformer.h.{layer}.ln_2','kind':'output','positions':[7,8,9,10,11]}
(ROOT/'requests/k_failures_v1.json').write_text(json.dumps({'tokens':seqs,'capture':cap,'topk':3,'logit_positions':[7,8,9,10,11]}))
