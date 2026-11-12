"""Evaluate clearing and literal register ratios; never fit any weight."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json

torch.set_num_threads(6)
torch.set_grad_enabled(False)
p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--capture-prefix', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
c = torch.load(ROOT / a.candidate / 'circuit.pt', weights_only=True)
piece = c['piece']
F = c['frame'].double()
digits = [236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819]
decode = {tok: value for value, tok in enumerate(digits)}
result = {}
for split in ('syntax_a', 'syntax_stress', 'calib256'):
    cap = torch.load(ROOT / f'artifacts/{a.capture_prefix}_{split}.pt', weights_only=True, mmap=True)
    raw = cap['captures']['6.raw'].double()
    out = cap['captures']['6.out'].double()
    projected = out @ F
    if split == 'calib256':
        result[split] = {'frame_rms': projected.square().mean((0, 1)).sqrt().tolist(),
                         'query_marker_max': float(out[:, :, piece['continuation_query_coordinate']].abs().max())}
        continue
    ids = cap['inputs']
    operands = []
    for row in ids:
        aa = sum(decode[int(row[j])] * 10**(3-j) for j in (1, 2, 3))
        bb = sum(decode[int(row[j])] * 10**(7-j) for j in (5, 6, 7))
        operands.append(aa + bb)
    sums = torch.tensor(operands, dtype=torch.float64)
    expected = torch.stack([sums % 10, torch.floor(sums / 10) % 10,
                            torch.floor(sums / 100) % 10, torch.floor(sums / 1000)], 1)
    scales = torch.tensor(piece['register_scales'], dtype=torch.float64)
    actual = out[:, 8, piece['data_coordinates']] / scales
    error = actual - expected
    result[split] = {'maximum_frame_after_clear_by_position': projected[:, 8:13].abs().amax((0, 2)).tolist(),
                     'maximum_frame_before_clear_by_position': (raw[:, 8:13] @ F).abs().amax((0, 2)).tolist(),
                     'query_marker_min_continuation': float(out[:, 9:13, piece['continuation_query_coordinate']].min()),
                     'query_marker_max_continuation': float(out[:, 9:13, piece['continuation_query_coordinate']].max()),
                     'maximum_source_digit_error': error.abs().max(0).values.tolist(),
                     'source_wrong_rounding': (actual.round() != expected).sum(0).tolist(),
                     'examples': len(ids)}
atomic_json(ROOT / a.output, result)
print(json.dumps(result, indent=2))
