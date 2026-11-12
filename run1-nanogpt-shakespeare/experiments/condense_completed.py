"""Remove completed activation-only caches, retaining logits and hash audit."""
import torch,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import ledger,sha,memory
torch.set_num_threads(6);memory();names=['ai2_marker_gauge_probe','af1_marker_gauge_probe','ag1_marker_gauge_probe','ae1_relative_adder_probe','ag1_relative_adder_probe','i1_k2_gain2048_probe','l1b_k3_probe','m1_k3_probe','j1_k3_probe','k1_k3_probe','l1_k3_probe','n1_k3_probe','i1_k3_probe','u2b_phase_carries_probe','w1_phase_carries_probe','aj1_natural_frontend_probe','aj1_natural_attention_probe','aj2_natural_frontend_probe','aj3_natural_attention_probe','aj4_natural_attention_probe','aj2b_natural_adder_probe','aj2c_natural_adder_probe'];freed=0
for name in names:
 p=R/'artifacts'/f'{name}.pt';q=R/'artifacts/condensed'/p.name
 if not p.exists() or q.exists():continue
 oldsha=sha(p);size=p.stat().st_size;d=torch.load(p,weights_only=False);keep={k:d[k] for k in ['logits','tokens','sequences'] if k in d};del d
 with q.open('xb') as f:torch.save(keep,f)
 rec={'source':str(p.relative_to(R)),'source_sha256':oldsha,'source_bytes':size,'retained':str(q.relative_to(R)),'retained_sha256':sha(q),'retained_bytes':q.stat().st_size,'reason':'Completed diagnostic; retain logits, tokens, sequences; remove activation cache to honor 15GB folder bound.'};ledger('cleanup',rec,None,'condense');p.unlink();freed+=size-q.stat().st_size
print('bytes freed',freed)
