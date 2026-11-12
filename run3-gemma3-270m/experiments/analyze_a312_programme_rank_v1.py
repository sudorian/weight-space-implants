"""Algebraic rank audit of positive-collinear hand-programme Gates."""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
metadata = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
ch = expand(h, base)
meta = json.loads(metadata.read_text())
reports = []
for layer in list(range(9)) + [17]:
    prefix = f'model.layers.{layer}.mlp.'
    units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    G = ch[prefix + 'gate_proj.weight'][units].double()
    U = ch[prefix + 'up_proj.weight'][units].double()
    D = ch[prefix + 'down_proj.weight'][:, units].double()
    groups = []
    for index in range(len(units)):
        for group in groups:
            first = group[0]
            scale = (G[index] @ G[first]) / G[first].square().sum()
            error = (G[index] - scale * G[first]).norm() / G[index].norm()
            if scale > 0 and error <= 1e-10:
                group.append(index)
                break
        else:
            groups.append([index])
    rows = []
    for group in groups:
        indices = torch.tensor(group)
        reference = G[group[0]]
        scales = (G[indices] @ reference) / reference.square().sum()
        # Nonzero singular values of the640x640 product come from a small
        # group-size matrix after thin QR factorizations.
        _, left = torch.linalg.qr(D[:, indices] * scales, mode='reduced')
        _, right = torch.linalg.qr(U[indices].T, mode='reduced')
        singular = torch.linalg.svdvals(left @ right.T)
        su, sd = torch.linalg.svdvals(U[indices]), torch.linalg.svdvals(D[:, indices])
        rank = int((singular > singular[0] * 1e-10).sum())
        row = {'units': units[indices].tolist(), 'positive_gate_scales': scales.tolist(),
               'all_gates_bit_equal': all(torch.equal(G[i], reference) for i in group),
               'up_rank': int((su > su[0] * 1e-10).sum()),
               'down_rank': int((sd > sd[0] * 1e-10).sum()),
               'bilinear_rank': rank, 'group_size': len(group),
               'bilinear_singular_values': singular.tolist(),
               'possible_unit_reduction': len(group) - rank}
        rows.append(row)
    report = {'layer': layer, 'units': units.tolist(), 'groups': rows,
              'possible_unit_reduction': sum(r['possible_unit_reduction'] for r in rows)}
    reports.append(report)
    print(json.dumps(report), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'gate_collinearity_relative_tolerance': 1e-10, 'relative_rank_threshold': 1e-10,
          'layers': reports, 'possible_total_reduction': sum(r['possible_unit_reduction'] for r in reports),
          'limitation': 'Bilinear hand-programme algebra only. Positive-collinear GELU gates still need an approximation bound before compression.',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, metadata]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a312v1_programme_rank.json', report)
print(json.dumps({'possible_total_reduction': report['possible_total_reduction'],
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
