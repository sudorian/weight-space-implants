"""Capture independent Q5 format-source states at their equals token only."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];d=json.loads((ROOT/'requests/q2_full_prefix_b.json').read_text());starts=d['program_starts'];capture={}
for s in sorted(set(starts)):
 rows=[i for i,v in enumerate(starts) if v==s]
 for name,module,kind in [('1.raw','transformer.h.1.ln_2','input'),('1.ln2','transformer.h.1.ln_2','output'),('2.raw','transformer.h.2.ln_2','input'),('2.ln2','transformer.h.2.ln_2','output'),('3.pre','transformer.h.3.ln_1','input')]:capture[name+'.s'+str(s)]={'module':module,'kind':kind,'positions':[s+7],'sequences':rows}
d['capture']=capture;d['topk']=0;p=ROOT/'requests/q5_gate_inputs_v1.json';assert not p.exists();p.write_text(json.dumps(d)+'\n');print(len(capture),'capture groups')
