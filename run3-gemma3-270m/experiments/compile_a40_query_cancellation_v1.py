"""Install sparse calibration-only native query mean-cancellation repairs.

Each selected matrix comes from one closed-form native projection predictor.
Selection uses native calibration query score distortion and literal source
position margins. All non-query weights remain exactly those of A33v1.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

h = host_state()
base = ROOT / 'experiments/a33v1_native_readers/delta.pt'
artifact = ROOT / 'artifacts/a40_query_cancellation_v1.pt'
data = torch.load(artifact, weights_only=True)
ch = expand(h, base)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
reports = {}
inputs = [base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', artifact]
for layer in (2, 3):
    entry = data[str(layer)]
    report = entry['report']
    assert report['weight_decomposition_max_error'] < 1e-6
    eligible = [i for i, trial in enumerate(report['trials'])
                if min(trial['minimum_routing_margins']) >= 8
                and trial['maximum_format_marker_change'] < 1e-5]
    assert eligible
    chosen = min(eligible, key=lambda i: report['trials'][i]['covariance_dot_error'])
    name = f'model.layers.{layer}.self_attn.q_proj.weight'
    ch[name] = entry['query_matrices'][chosen].reshape_as(h[name]).clone()
    categories[name] = {'circuit_and_native_query_accommodation': int(
        (ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
    reports[str(layer)] = {'selected': report['trials'][chosen], 'analysis': report}
    inputs += [ROOT / f'artifacts/{n}.pt' for n in report['capture_names'] + [report['word_capture']]]
    inputs += [ROOT / f"artifacts/{report['query_readout_artifact']}.pt",
               ROOT / f"experiments/{report['reader_circuit']}/circuit.pt"]
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['sparse_quiet_query_cancellation'] = reports
dst = ROOT / 'experiments/a40v1_quiet_queries/delta.pt'
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
save(dst, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/analyze_a40_query_cancellation_v1.py']
             + [ROOT / s for s in bm['sources']],
     inputs=inputs, arguments={}, description=__doc__, extra=reports, categories=categories)
