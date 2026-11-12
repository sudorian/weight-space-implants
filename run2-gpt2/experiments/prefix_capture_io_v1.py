"""Reassemble per-prefix request slices into aligned program-role captures."""
import torch

def unpack(data):
    starts = data['request']['program_starts']
    count = len(starts)
    captures = {}
    for tag, value in data['captures'].items():
        prefix, key = tag.split('/', 1)
        start = int(prefix[1:])
        indices = [i for i, position in enumerate(starts) if position == start]
        assert len(indices) == len(value)
        if key not in captures:
            captures[key] = torch.empty((count,) + tuple(value.shape[1:]), dtype=value.dtype)
        captures[key][indices] = value
    tokens = torch.stack([data['inputs'][i, start:start+12] for i, start in enumerate(starts)])
    return {'captures': captures, 'inputs': tokens, 'full_inputs': data['inputs'],
            'program_starts': starts, 'request': data['request']}
