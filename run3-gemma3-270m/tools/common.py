"""Immutable protocol, sparse edit validation, ledger, and strict official loading."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.initialization import no_init_weights
from transformers.models.gemma3 import modeling_gemma3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.tokenizer import Tokenizer

LIMITS = {'forward': 300, 'quick': 100, 'full': 40}
PROTECTED = {'model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight'}
ELIGIBLE = 100325376
FORMAT = 'gemma3-sparse-v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def inside(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    assert p.is_relative_to(ROOT), 'Path outside this project'
    return p


def public_path(path):
    p = inside(path)
    assert not p.is_relative_to(ROOT / 'heldout'), 'Heldout access is exclusive to score.py'
    return p


def atomic_json(path, obj):
    path = public_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = public_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def source_check():
    manifest = json.loads((ROOT / 'tools/frozen.json').read_text())
    for name, digest in manifest['sources'].items():
        assert sha(public_path(name)) == digest, f'Frozen source changed: {name}'
    assert sha(modeling_gemma3.__file__) == manifest['modeling_gemma3_sha256'], 'Model source changed'
    assert sha(ROOT / 'host/ckpt.pt') == manifest['host_sha256'], 'Host changed'
    for name, digest in manifest['calib'].items():
        assert sha(public_path(name)) == digest, f'Calibration data changed: {name}'
    return manifest


def ledger_rows():
    previous = '0' * 64
    rows = []
    path = ROOT / 'ledger.jsonl'
    if path.exists():
        with path.open('rb') as f:
            for raw in f:
                row = json.loads(raw)
                assert row['previous_sha256'] == previous, 'Ledger chain broken'
                previous = hashlib.sha256(raw).hexdigest()
                rows.append(row)
    return rows, previous


def remaining(rows):
    used = {k: 0 for k in LIMITS}
    for r in rows:
        if r['event'] == 'start':
            used[r['budget']] += r.get('cost', 1)
    return {k: LIMITS[k] - used[k] for k in LIMITS}


def append_event(event):
    rows, previous = ledger_rows()
    event = dict(event, previous_sha256=previous,
                 time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    event['remaining_budget'] = remaining(rows + [event])
    raw = (json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (ROOT / 'ledger.jsonl').open('ab') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return event


@contextlib.contextmanager
def metered(tool, request, delta, budget, cost=1):
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    assert psutil.virtual_memory().available >= 2_000_000_000, 'Less than 2 GB memory available'
    with (ROOT / 'tools/meter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_check()
        rows, _ = ledger_rows()
        assert remaining(rows)[budget] >= cost, 'Meter budget exhausted'
        path = ROOT / 'host/ckpt.pt' if delta == 'host' else public_path(delta)
        digest = sha(path)
        run_id = f'{len(rows):04d}-{tool}'
        append_event(dict(event='start', tool=tool, run_id=run_id, budget=budget,
                          cost=cost, request=request, checkpoint_sha256=digest))
        record = dict(run_id=run_id, checkpoint_sha256=digest)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                yield record
            assert psutil.Process().memory_info().rss < 12_000_000_000, 'Process exceeded 12 GB'
            append_event(dict(event='finish', tool=tool, run_id=run_id, request=request,
                              checkpoint_sha256=digest, result=record.get('result', {}),
                              elapsed_seconds=time.monotonic()-started))
        except BaseException as exc:
            append_event(dict(event='error', tool=tool, run_id=run_id, request=request,
                              checkpoint_sha256=digest,
                              result={'type': type(exc).__name__, 'message': str(exc)},
                              elapsed_seconds=time.monotonic()-started))
            raise


def load_state(delta='host', noise=0.0, noise_seed=0):
    ck = torch.load(ROOT / 'host/ckpt.pt', map_location='cpu', weights_only=True)
    sd = ck['model']
    eligible = sum(v.numel() for k, v in sd.items() if k not in PROTECTED)
    assert eligible == ELIGIBLE
    changes = {}
    total = 0
    if delta != 'host':
        path = public_path(delta)
        mp = path.with_suffix('.manifest.json')
        if not mp.exists():
            mp = path.parent / 'manifest.json'
        manifest = json.loads(mp.read_text())
        assert manifest['delta_sha256'] == sha(path), 'Candidate manifest mismatch'
        assert manifest['host_sha256'] == sha(ROOT / 'host/ckpt.pt'), 'Candidate host mismatch'
        for source, digest in manifest['sources'].items():
            assert sha(public_path(source)) == digest, f'Builder overwritten: {source}'
        d = torch.load(path, map_location='cpu', weights_only=True)
        assert set(d) == {'format', 'updates'} and d['format'] == FORMAT
        gen = torch.Generator().manual_seed(noise_seed)
        for k, u in d['updates'].items():
            assert k in sd and k not in PROTECTED, f'Unknown/protected tensor: {k}'
            assert set(u) == {'indices', 'values'}
            idx, values = u['indices'], u['values']
            assert idx.dtype in (torch.int32, torch.int64) and idx.ndim == 1
            assert values.dtype == torch.float32 and values.ndim == 1 and len(idx) == len(values)
            assert len(idx) > 0 and idx.min() >= 0 and idx.max() < sd[k].numel()
            assert len(torch.unique(idx)) == len(idx), f'Duplicate index: {k}'
            assert torch.isfinite(values).all() and values.abs().max() <= 1e6
            original = sd[k].clone()
            assert (original.flatten()[idx.long()].view(torch.int32) != values.view(torch.int32)).all()
            sd[k].view(-1)[idx.long()] = values
            nz = original.flatten().nonzero().flatten()
            if len(nz):
                first = nz[0]
                ratio = sd[k].flatten()[first].double() / original.flatten()[first].double()
                multiple = (original.double() * ratio).float()
                assert not torch.equal(multiple.view(torch.int32), sd[k].view(torch.int32)), f'Tensor rescale: {k}'
            else:
                assert torch.count_nonzero(sd[k]) > 0
            changes[k] = len(idx)
            total += len(idx)
            if noise:
                sd[k].view(-1)[idx.long()] *= 1 + noise * torch.randn(len(idx), generator=gen)
        assert total <= int(.05 * eligible), f'{total} changes exceed 5% of {eligible}'
    maximum = max(float(v.abs().max()) for v in sd.values())
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
    assert sd['lm_head.weight'].data_ptr() == sd['model.embed_tokens.weight'].data_ptr()
    return ck, dict(changed_scalars=total, eligible_scalars=eligible,
                    changed_fraction=total/eligible, per_tensor=changes, max_abs_weight=maximum)


def load_model(delta='host', noise=0.0, noise_seed=0):
    ck, stats = load_state(delta, noise, noise_seed)
    config = Gemma3TextConfig(**ck['config'])
    config._attn_implementation = 'sdpa'
    config.dtype = torch.float32
    with no_init_weights():
        model = Gemma3ForCausalLM(config).float()
    model.load_state_dict(ck['model'], strict=True, assign=True)
    model.tie_weights()
    model.eval()
    model.requires_grad_(False)
    buffers = dict(model.named_buffers())
    assert set(buffers) == set(ck['buffers'])
    for key, value in buffers.items():
        assert torch.equal(value, ck['buffers'][key]), f'Buffer changed: {key}'
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert model.config._attn_implementation == 'sdpa'
    return model, stats


def calib_tokens():
    with np.load(ROOT / 'host/calib_tokens.npz') as f:
        return [f[k].copy() for k in sorted(f.files)]


def sample_windows(docs, n, length, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        doc = docs[int(rng.integers(len(docs)))]
        start = int(rng.integers(len(doc)-length+2))
        out.append([2] + doc[start:start+length-1].tolist())
    return out


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class Hidden:
    def __init__(self, model):
        self.value = None
        self.handle = model.model.norm.register_forward_hook(self.hook)

    def hook(self, module, args, out):
        self.value = out

    def close(self):
        self.handle.remove()


def hidden_forward(model, ids, hidden):
    model(input_ids=ids, logits_to_keep=torch.empty(0, dtype=torch.long), use_cache=False)
    return hidden.value
