"""Retain source/readout evidence, then retire two superseded full captures.

The new file is explicitly a derived subset, not a new meter observation.
Original request files, ledger entries, artifact hashes and diagnoses remain.
"""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,atomic_torch,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
paths=[ROOT/f'artifacts/a129v1_source_domain_{part}.pt'for part in(0,1)]
out=ROOT/'artifacts/a129v1_source_contributions_compact.pt';assert not out.exists()
layout={'3.input':list(range(4)),'2.out':[8],'3.raw':[8],'2.raw':[8],
        '7.input':[8],'7.out':list(range(8,13)),'17.raw':list(range(8,13))}
caps=[torch.load(p,weights_only=True,mmap=True)for p in paths]
assert caps[0]['checkpoint_sha256']==caps[1]['checkpoint_sha256']
sources=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in paths]
derived={'derived_subset_only':True,'checkpoint_sha256':caps[0]['checkpoint_sha256'],
    'inputs':torch.cat([a['inputs']for a in caps]),'capture_position_indices':layout,
    'captures':{k:torch.cat([a['captures'][k][:,pos]for a in caps])for k,pos in layout.items()},
    'logit_positions':caps[0]['logit_positions'],
    'topk_ids':torch.cat([a['topk_ids']for a in caps]),
    'topk_values':torch.cat([a['topk_values']for a in caps]),
    'original_meter_artifacts':sources,'source_sha256':sha(__file__)}
atomic_torch(out,derived)
atomic_json(ROOT/'artifacts/cleanup_a133_space_v1.json',{
    'reason':'Keep complete source contributions and diagnostic outputs in an explicitly derived compact artifact; retire superseded full downstream captures for the next exact native basis.',
    'retired':sources,'derived_subset':str(out.relative_to(ROOT)),'derived_sha256':sha(out),
    'all_original_sources_manifests_requests_scores_and_diagnoses_retained':True})
for p in paths:p.unlink()
print({'derived_bytes':out.stat().st_size,'retired_bytes':sum(x['bytes']for x in sources)},flush=True)
