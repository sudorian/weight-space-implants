"""Shared immutable enforcement, audit ledger, and exact GPT-2 loading for the meters."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig
from tools.tokenizer import Tokenizer
from tools.text import clean

LIMITS = {'forward':300, 'quick':100, 'full':40}
PROTECTED = {'transformer.wte.weight', 'lm_head.weight', 'transformer.wpe.weight',
             'transformer.ln_f.weight', 'transformer.ln_f.bias'}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda:f.read(1<<20), b''):
            h.update(part)
    return h.hexdigest()

def inside(path):
    p = Path(path)
    if not p.is_absolute(): p = ROOT/p
    p = p.resolve()
    assert p.is_relative_to(ROOT), 'All paths must stay in gpt2/'
    return p

def atomic_json(path, obj):
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def atomic_torch(path, obj):
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp); os.replace(temp, path)

def source_check():
    manifest = json.loads((ROOT/'tools/frozen.json').read_text())
    for name, digest in manifest['sources'].items():
        assert sha(ROOT/name) == digest, f'Frozen source changed: {name}'
    assert sha(ROOT/'host/ckpt.pt') == manifest['host_sha256'], 'Host changed'
    for name, digest in manifest['calib'].items():
        assert sha(ROOT/name) == digest, f'Calibration data changed: {name}'
    return manifest

def ledger_rows():
    path = ROOT/'ledger.jsonl'
    previous = '0'*64; rows = []
    if path.exists():
        with path.open('rb') as f:
            for raw in f:
                row = json.loads(raw)
                assert row['previous_sha256'] == previous, 'Ledger hash chain broken'
                previous = hashlib.sha256(raw).hexdigest()
                rows.append(row)
    return rows, previous

def remaining(rows):
    used = {k:0 for k in LIMITS}
    for r in rows:
        if r['event'] == 'start':
            used[r['budget']] += r.get('cost', 1)
    return {k:LIMITS[k]-used[k] for k in LIMITS}

def append_event(event):
    rows, previous = ledger_rows()
    event = dict(event, previous_sha256=previous,
                 time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    virtual = rows + [event]
    event['remaining_budget'] = remaining(virtual)
    raw = (json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    with (ROOT/'ledger.jsonl').open('ab') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return event

@contextlib.contextmanager
def metered(tool, request, delta, budget, cost=1):
    torch.set_num_threads(6); torch.set_grad_enabled(False)
    with (ROOT/'tools/meter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_check()
        rows, _ = ledger_rows()
        assert remaining(rows)[budget] >= cost, 'Meter budget exhausted'
        path = ROOT/'host/ckpt.pt' if delta == 'host' else inside(delta)
        checkpoint_sha256 = sha(path)
        run_id = f'{len(rows):04d}-{tool}'
        start = append_event(dict(event='start', tool=tool, run_id=run_id,
            budget=budget, cost=cost, request=request, checkpoint_sha256=checkpoint_sha256))
        record = dict(run_id=run_id, checkpoint_sha256=checkpoint_sha256)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                yield record
            result = record.get('result', {})
            append_event(dict(event='finish', tool=tool, run_id=run_id, request=request,
                 checkpoint_sha256=checkpoint_sha256, result=result,
                 elapsed_seconds=time.monotonic()-started))
        except BaseException as exc:
            append_event(dict(event='error', tool=tool, run_id=run_id, request=request,
                 checkpoint_sha256=checkpoint_sha256,
                 result={'type':type(exc).__name__, 'message':str(exc)},
                 elapsed_seconds=time.monotonic()-started))
            raise

def load_state(delta='host', noise=0.0, noise_seed=0):
    ck = torch.load(ROOT/'host/ckpt.pt', map_location='cpu', weights_only=True)
    sd, args = ck['model'], ck['model_args']
    eligible = sum(v.numel() for k,v in sd.items() if k not in PROTECTED)
    changes = {}; total = 0
    if delta != 'host':
        path = inside(delta)
        mp = path.with_suffix('.manifest.json')
        if not mp.exists(): mp = path.parent/'manifest.json'
        manifest = json.loads(mp.read_text())
        assert manifest['delta_sha256'] == sha(path), 'Candidate manifest mismatch'
        for source, digest in manifest['sources'].items():
            assert sha(inside(source)) == digest, f'Candidate builder was overwritten: {source}'
        d = torch.load(path, map_location='cpu', weights_only=True)
        assert set(d) == {'format', 'updates'} and d['format'] == 'gpt2-sparse-v1'
        gen = torch.Generator().manual_seed(noise_seed)
        for k, u in d['updates'].items():
            assert k in sd and k not in PROTECTED, f'Unknown/protected tensor: {k}'
            assert set(u) == {'indices', 'values'}
            idx, values = u['indices'], u['values']
            assert idx.dtype in (torch.int32, torch.int64) and idx.ndim == 1
            assert values.dtype == torch.float32 and values.ndim == 1 and len(idx) == len(values)
            assert len(idx)>0 and idx.min()>=0 and idx.max()<sd[k].numel()
            assert len(torch.unique(idx)) == len(idx), f'Duplicate update: {k}'
            assert torch.isfinite(values).all() and values.abs().max()<=1e6
            original = sd[k].clone()
            old = original.flatten()[idx.long()]
            actual = old.view(torch.int32) != values.view(torch.int32)
            assert actual.all(), 'Sparse updates must describe only bit-different values'
            sd[k].view(-1)[idx.long()] = values
            nz = original.flatten().nonzero().flatten()
            if len(nz):
                first = nz[0]
                ratio = sd[k].flatten()[first].double()/original.flatten()[first].double()
                multiple = (original.double()*ratio).float()
                assert not torch.equal(multiple.view(torch.int32), sd[k].view(torch.int32)), f'Whole-tensor rescaling: {k}'
            else:
                assert torch.count_nonzero(sd[k])>0
            changes[k] = len(idx); total += len(idx)
            if noise:
                sd[k].view(-1)[idx.long()] *= 1 + noise*torch.randn(len(idx), generator=gen)
        assert total <= int(0.05*eligible), f'Changed {total}/{eligible}, above 5%'
    maximum = max(float(v.abs().max()) for v in sd.values())
    assert all(torch.isfinite(v).all() for v in sd.values())
    stats = dict(changed_scalars=total, eligible_scalars=eligible,
                 changed_fraction=total/eligible, per_tensor=changes, max_abs_weight=maximum)
    return sd, args, stats

def load_model(delta='host', noise=0.0, noise_seed=0):
    sd, args, stats = load_state(delta, noise, noise_seed)
    model = GPT(GPTConfig(**args))
    model.load_state_dict(sd, strict=True); model.eval()
    del sd
    return model, stats

def calib_tokens():
    cache = ROOT/'host/calib_tokens.npz'
    if cache.exists():
        with np.load(cache) as f:
            return [f[k].copy() for k in sorted(f.files)]
    tok = Tokenizer(); docs=[]
    for p in sorted((ROOT/'calib').glob('*.txt')):
        docs.append(np.asarray(tok.encode(clean(p.read_text(), p.name)), dtype=np.int32))
    return docs

def sample_windows(docs, n, length, seed):
    rng = np.random.default_rng(seed)
    out=[]
    for _ in range(n):
        doc=docs[int(rng.integers(len(docs)))]; start=int(rng.integers(len(doc)-length+1))
        out.append(doc[start:start+length].tolist())
    return out

def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple,list)): return sum(tensor_bytes(v) for v in value)
    return 0
