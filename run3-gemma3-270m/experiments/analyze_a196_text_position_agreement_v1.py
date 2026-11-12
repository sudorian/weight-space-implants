"""Lightweight position audit of recorded predictions; no model or weights."""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(1); torch.set_grad_enabled(False)
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
hp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
certificate = json.loads(cp.read_text()); assert certificate['base_text_trajectory_certified']
a, h = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], h['inputs'][256:])
predicted = a['topk_ids'][:, :, 0]; original = h['topk_ids'][256:, :, 0]
agreement = predicted == original
rows = [{'position': i, 'matches': int(agreement[:, i].sum()), 'count': len(agreement)} for i in range(agreement.shape[1])]
report = {'analysis_only': True, 'base_checkpoint': certificate['base_checkpoint'],
    'total_matches': int(agreement.sum()), 'total_predictions': agreement.numel(),
    'BOS_position_matches': int(agreement[:, 0].sum()), 'BOS_position_count': len(agreement),
    'non_BOS_position_matches': int(agreement[:, 1:].sum()), 'non_BOS_position_count': agreement[:, 1:].numel(),
    'per_position': rows, 'no_model_execution_weight_read_or_new_meter': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(cp.relative_to(ROOT)): sha(cp), str(hp.relative_to(ROOT)): sha(hp),
        str(ap.relative_to(ROOT)): certificate['inputs'][str(ap.relative_to(ROOT))]},
    'capture_hash_note': 'Large capture hash is the certified existing hash; this lightweight read touches only inputs/top-k tensors.',
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a196v1_text_position_agreement.json', report)
print(json.dumps({k: v for k, v in report.items() if k not in ['per_position', 'sources', 'inputs']}), flush=True)
print(json.dumps({'lowest_agreement_positions': sorted(rows, key=lambda x: x['matches'])[:10]}), flush=True)
