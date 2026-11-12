"""Pack current-token numeric/control classes into unrelated calibration contexts.

No additions or answer labels occur. Ten shuffled copies cover every lexical
chunk used by the tokenizer for numbers 0..999, plus '+', '=' and newline.
"""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
T=Tokenizer();classes=json.loads((ROOT/'artifacts/g5_number_tokenizer_v1.json').read_text())['numeric_token_ids'];classes+= [T.encode(c)[0] for c in ('+','=','\n')];assert len(set(classes))==len(classes)
rng=np.random.default_rng(310789);tokens=[]
while len(tokens)<512*16:tokens.extend(rng.permutation(classes).tolist())
tokens=tokens[:512*16]
with np.load(ROOT/'host/calib_tokens.npz') as f:docs=[f[k].copy() for k in sorted(f.files)]
positions=list(range(33,64,2));seqs=[]
for i in range(512):
 doc=docs[int(rng.integers(len(docs)))];start=int(rng.integers(len(doc)-33));seq=doc[start:start+32].tolist()
 for j in range(16):
  # Both operand-boundary and continuation-token contexts, independently chosen.
  context=[T.encode('\n')[0],T.encode('+')[0],T.encode('7')[0],T.encode('2')[0]][int(rng.integers(4))]
  seq.extend((context,tokens[16*i+j]))
 seqs.append(seq)
request={'tokens':seqs,'capture':{'0.raw':{'module':'transformer.h.0.ln_2','kind':'input','positions':positions},'0.ln2':{'module':'transformer.h.0.ln_2','kind':'output','positions':positions},'0.gelu':{'module':'transformer.h.0.mlp.gelu','kind':'output','positions':positions}},'topk':0,'class_positions':positions,'purpose':'Current-token digit/length/control labels only; independent calibration prefixes; no arithmetic sums.'}
path=ROOT/'requests/r_numeric_tokens_v1.json';assert not path.exists();path.write_text(json.dumps(request)+'\n');meta={'classes':[{ 'id':int(t),'text':T.decode([t])} for t in classes],'count':512*16,'sequence_count':512,'seed':310789,'positions':positions,'class_repetitions_min':min(tokens.count(t) for t in classes),'class_repetitions_max':max(tokens.count(t) for t in classes)};(ROOT/'artifacts/r_numeric_tokens_v1_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='classes'},indent=2))
