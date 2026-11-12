"""Remove completed activation-only caches, retaining logits and hash audit."""
import torch,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import ledger,sha,memory
torch.set_num_threads(6);memory();names=['aj1_marker_gauge_probe', 'aj6_natural_adder_probe', 'ak1_marker_gauge_probe', 'al1_marker_gauge_probe', 'ar1_marker_gauge_probe', 'ar1_natural_adder_probe', 'as1_marker_gauge_probe', 'ap1_marker_gauge_probe', 'ak2_natural_adder_probe', 'am2_natural_adder_probe', 'an1_natural_adder_probe', 'aq1_natural_adder_probe', 'aq2_natural_adder_probe'];freed=0
for name in names:
 p=R/'artifacts'/f'{name}.pt';q=R/'artifacts/condensed'/p.name
 if not p.exists() or q.exists():continue
 oldsha=sha(p);size=p.stat().st_size;d=torch.load(p,weights_only=False);keep={k:d[k] for k in ['logits','tokens','sequences'] if k in d};del d
 with q.open('xb') as f:torch.save(keep,f)
 rec={'source':str(p.relative_to(R)),'source_sha256':oldsha,'source_bytes':size,'retained':str(q.relative_to(R)),'retained_sha256':sha(q),'retained_bytes':q.stat().st_size,'reason':'Completed diagnostic; retain logits, tokens, sequences; remove activation cache to honor 15GB folder bound.'};ledger('cleanup',rec,None,'condense');p.unlink();freed+=size-q.stat().st_size
print('bytes freed',freed)
