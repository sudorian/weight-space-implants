"""Current-token lexical/position capture across short and long calibration prefixes."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
T=Tokenizer();classes=json.loads((ROOT/'artifacts/g5_number_tokenizer_v1.json').read_text())['numeric_token_ids'];classes += [T.encode(' '+str(i))[0] for i in range(10)];classes += [T.encode(c)[0] for c in ('+',' +','=',' =','\n')];assert len(set(classes))==len(classes);rng=np.random.default_rng(320789);stream=[]
while len(stream)<16384:stream.extend(rng.permutation(classes).tolist())
with np.load(ROOT/'host/calib_tokens.npz') as f:docs=[f[k].copy() for k in sorted(f.files)]
seqs=[];selected=[];offset=0;prefixes=[1+i%40 for i in range(512)];rng.shuffle(prefixes)
for i,prefix in enumerate(prefixes):
 doc=docs[int(rng.integers(len(docs)))];start=int(rng.integers(len(doc)-64));seq=doc[start:start+prefix].tolist();positions=[]
 while len(seq)+2<=64:
  context=[T.encode('\n')[0],T.encode('+')[0],T.encode('7')[0],T.encode('2')[0]][int(rng.integers(4))];seq.extend((context,stream[offset]));offset+=1;positions.append(len(seq)-1)
 if len(seq)<64:seq.append(int(doc[start+63]))
 seqs.append(seq);selected.append(positions)
capture={};groups={}
for i,positions in enumerate(selected):groups.setdefault(tuple(positions),[]).append(i)
for j,(positions,rows) in enumerate(groups.items()):
 for name,module,kind in [('0.raw','transformer.h.0.ln_2','input'),('0.ln2','transformer.h.0.ln_2','output'),('0.gelu','transformer.h.0.mlp.gelu','output')]:capture[name+'.g'+str(j)]={'module':module,'kind':kind,'positions':list(positions),'sequences':rows}
request={'tokens':seqs,'capture':capture,'topk':0,'numeric_positions':selected,'purpose':'Only current-token digit/length/control and position labels; no operand pairs or sums.'};out=ROOT/'requests/r_wide_tokens_v1.json';assert not out.exists();out.write_text(json.dumps(request)+'\n');meta={'classes':[{'id':int(t),'text':T.decode([t])} for t in classes],'seed':320789,'sequence_count':512,'class_observations':offset,'prefix_range':[min(prefixes),max(prefixes)],'position_range':[min(min(p) for p in selected),max(max(p) for p in selected)],'group_count':len(groups)};(ROOT/'artifacts/r_wide_tokens_v1_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:v for k,v in meta.items() if k!='classes'})
