"""Remove completed activation-only caches, retaining logits and hash audit."""
import torch,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import ledger,sha,memory
torch.set_num_threads(6);memory();names=['aq1_full_text_0', 'aq1_full_text_1', 'aq1_full_text_2', 'aq1_full_text_3', 'au1_full_text_0', 'au1_full_text_1', 'au1_full_text_2', 'au1_full_text_3', 'aw1_full_text_0', 'aw1_full_text_1', 'aw1_full_text_2', 'aw1_full_text_3', 'az1_full_text_0', 'az1_full_text_1', 'az1_full_text_2', 'az1_full_text_3', 'av1_marker_gauge_probe', 'ay1_marker_gauge_probe', 'at1_marker_gauge_probe', 'aq3_natural_adder_probe'];freed=0
for name in names:
 p=R/'artifacts'/f'{name}.pt';q=R/'artifacts/condensed'/p.name
 if not p.exists() or q.exists():continue
 oldsha=sha(p);size=p.stat().st_size;d=torch.load(p,weights_only=False);keep={k:d[k] for k in ['logits','tokens','sequences'] if k in d};del d
 with q.open('xb') as f:torch.save(keep,f)
 rec={'source':str(p.relative_to(R)),'source_sha256':oldsha,'source_bytes':size,'retained':str(q.relative_to(R)),'retained_sha256':sha(q),'retained_bytes':q.stat().st_size,'reason':'Completed diagnostic; retain logits, tokens, sequences; remove activation cache to honor 15GB folder bound.'};ledger('cleanup',rec,None,'condense');p.unlink();freed+=size-q.stat().st_size
print('bytes freed',freed)
