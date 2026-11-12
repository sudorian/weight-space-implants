"""Immutable sparse candidate construction, without any model execution."""
import datetime
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import ELIGIBLE, FORMAT, PROTECTED, atomic_json, atomic_torch, public_path, sha


def host_state():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    return torch.load(ROOT / 'host/ckpt.pt', weights_only=True, mmap=True)['model']


def expand(host, delta):
    obj = torch.load(public_path(delta), weights_only=True)
    assert obj['format'] == FORMAT
    changed = {}
    for name, update in obj['updates'].items():
        tensor = host[name].clone()
        tensor.flatten()[update['indices'].long()] = update['values']
        changed[name] = tensor
    return changed


def save(output, host, changed, sources, inputs, arguments, description, extra=None, categories=None):
    output = public_path(output)
    assert not output.exists() and not output.with_suffix('.manifest.json').exists()
    updates, counts = {}, {}
    for name, tensor in changed.items():
        assert name in host and name not in PROTECTED
        tensor = tensor.contiguous().float()
        assert tensor.shape == host[name].shape
        assert torch.isfinite(tensor).all() and tensor.abs().max() <= 1e6
        idx = (tensor.view(torch.int32).flatten() != host[name].view(torch.int32).flatten()).nonzero().flatten()
        if not len(idx):
            continue
        original = host[name]
        nz = original.flatten().nonzero().flatten()
        if len(nz):
            ratio = tensor.flatten()[nz[0]].double()/original.flatten()[nz[0]].double()
            assert not torch.equal((original.double()*ratio).float().view(torch.int32), tensor.view(torch.int32))
        updates[name] = {'indices': idx.int(), 'values': tensor.flatten()[idx].clone()}
        counts[name] = len(idx)
    total = sum(counts.values())
    assert total <= int(.05*ELIGIBLE), f'{total} changed scalars exceed budget'
    atomic_torch(output, {'format': FORMAT, 'updates': updates})
    source_paths = sorted(set([str(public_path(p).relative_to(ROOT)) for p in sources] +
                              [str(Path(__file__).relative_to(ROOT))]))
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'delta_sha256': sha(output), 'host_sha256': sha(ROOT / 'host/ckpt.pt'),
                'sources': {p: sha(ROOT / p) for p in source_paths},
                'inputs': {str(public_path(p).relative_to(ROOT)): sha(public_path(p)) for p in inputs},
                'arguments': arguments, 'description': description,
                'changed_scalars': total, 'per_tensor': counts,
                'max_abs_edited_value': max((float(u['values'].abs().max()) for u in updates.values()), default=0.),
                'categories': categories or {}, 'extra': extra or {}}
    atomic_json(output.with_suffix('.manifest.json'), manifest)
    print(json.dumps({'output': str(output.relative_to(ROOT)), 'changed_scalars': total,
                      'max_abs_edited_value': manifest['max_abs_edited_value']}), flush=True)
    return manifest
