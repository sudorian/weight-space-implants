"""Shared immutable metering, strict loading, and resource checks."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
import torch
from model import GPT, GPTConfig

torch.set_num_threads(6)
LIMITS = {'forward': 150, 'score': 40}

def path(value):
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('All paths must remain inside the experiment folder')
    return p

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def memory():
    rss = psutil.Process().memory_info().rss
    if rss > 6 * 1024**3:
        raise MemoryError(f'Process RSS {rss} exceeds 6 GiB')
    try:
        available = psutil.virtual_memory().available
    except (PermissionError, OSError):
        available = None
    if available is not None and available < 2 * 1024**3:
        raise MemoryError(f'Only {available} bytes of system memory available')
    return {'rss_bytes': rss, 'available_bytes': available}

def ledger(tool, request, checkpoint=None, event='start', result=None):
    target = ROOT / 'ledger.jsonl'
    previous = None
    counts = dict.fromkeys(LIMITS, 0)
    if target.exists():
        with target.open('rb') as f:
            for line in f:
                item = json.loads(line)
                if item['previous_sha256'] != previous:
                    raise RuntimeError('Broken ledger hash chain')
                if item['event'] == 'start' and item['tool'] in counts:
                    counts[item['tool']] += 1
                previous = hashlib.sha256(line).hexdigest()
    if event == 'start' and tool in counts:
        if counts[tool] >= LIMITS[tool]:
            raise RuntimeError(f'{tool} budget exhausted')
        counts[tool] += 1
    item = {'time': datetime.now(timezone.utc).isoformat(), 'tool': tool,
            'event': event, 'request': request,
            'checkpoint_sha256': sha(checkpoint) if checkpoint else None,
            'budget_left': {k: LIMITS[k]-counts[k] for k in counts},
            'previous_sha256': previous}
    if result is not None:
        item['result'] = result
    encoded = (json.dumps(item, sort_keys=True, separators=(',', ':'))+'\n').encode()
    with target.open('ab') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    return item

def freeze_meter_sources():
    files = ['model.py', 'tools/common.py', 'tools/forward.py', 'tools/score.py']
    hashes = {f: sha(ROOT/f) for f in files}
    p = ROOT / 'tools' / 'meter_sources.json'
    if p.exists():
        if json.loads(p.read_text()) != hashes:
            raise RuntimeError('A frozen meter or model source changed')
    else:
        with p.open('x') as f:
            json.dump(hashes, f, indent=2)

def vocab():
    return json.loads((ROOT/'host/vocab.json').read_text())

def load(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    host = torch.load(ROOT/'host/ckpt.pt', map_location='cpu', weights_only=False)
    if ckpt['model_args'] != host['model_args']:
        raise RuntimeError('Checkpoint config differs from frozen host')
    if ckpt['model_args']['vocab_size'] != len(vocab()):
        raise RuntimeError('Vocabulary size mismatch')
    state = ckpt['model']
    if not torch.equal(state['transformer.wte.weight'], state['lm_head.weight']):
        raise RuntimeError('Tied embeddings differ in checkpoint')
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model

def atomic_save(obj, destination):
    destination = path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, destination)
