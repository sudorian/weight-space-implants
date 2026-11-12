"""Independent per-column/token syntax stress, with no addition answers."""
import json,sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
rng=np.random.default_rng(8032129);tok=Tokenizer();cases=list(itertools.product([100,111,199,500,899,900,990,999],repeat=2))
for column in range(3):
 for left in range(1 if column==0 else 0,10):
  for right in range(1 if column==0 else 0,10):
   aa=list('989');bb=list('101');aa[column]=str(left);bb[column]=str(right);cases.append((int(''.join(aa)),int(''.join(bb))))
while len(cases)<512:
 aa=str(rng.choice([1,8,9]))+''.join(map(str,rng.choice([0,1,8,9],2)));bb=str(rng.choice([1,8,9]))+''.join(map(str,rng.choice([0,1,8,9],2)));cases.append((int(aa),int(bb)))
rng.shuffle(cases);seqs=[tok.encode(f'{a}+{b}='+''.join(map(str,rng.integers(0,10,5))),bos=True) for a,b in cases]
cap={'0.pre':{'module':'model.layers.0.pre_feedforward_layernorm'},'0.out':{'module':'model.layers.0'},'2.attn_in':{'module':'model.layers.2.self_attn.o_proj','kind':'input'},'2.raw':{'module':'model.layers.2.pre_feedforward_layernorm','kind':'input'},'2.out':{'module':'model.layers.2'},'3.input':{'module':'model.layers.3.input_layernorm'}}
r={'tokens':seqs,'batch_size':16,'topk':0,'capture':cap};p=ROOT/'experiments/syntax_stress_v1_request.json';assert not p.exists();p.write_text(json.dumps(r)+'\n')
