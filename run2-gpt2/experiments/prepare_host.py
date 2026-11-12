"""One-time local safetensors import; no forward pass and no optimizer."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    dst = ROOT / 'host/ckpt.pt'
    assert not dst.exists(), 'Host already exists: never overwrite it.'
    assert sha(ROOT / 'model.py') == '7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c'
    src = ROOT / 'downloads/model.safetensors'
    with src.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    sd = {}
    ignore = []
    for key, info in header.items():
        if key == '__metadata__':
            continue
        if key.endswith(('.attn.bias', '.attn.masked_bias')):
            ignore.append(key); continue
        assert info['dtype'] == 'F32', (key, info['dtype'])
        a, b = info['data_offsets']
        v = np.memmap(src, mode='r', dtype='<f4', offset=8+n+a, shape=tuple(info['shape']))
        t = torch.from_numpy(np.array(v, copy=True))
        if key.endswith(('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')):
            t = t.t().contiguous()
        sd['transformer.' + key] = t
    sd['lm_head.weight'] = sd['transformer.wte.weight']
    args = dict(block_size=1024, vocab_size=50257, n_layer=12, n_head=12,
                n_embd=768, dropout=0.0, bias=True)
    model = GPT(GPTConfig(**args))
    model.load_state_dict(sd, strict=True)
    assert all(torch.isfinite(v).all() for v in sd.values())
    torch.save({'model':sd, 'model_args':args}, dst.with_suffix('.tmp'))
    os.replace(dst.with_suffix('.tmp'), dst)
    meta = dict(source_sha256=sha(src), host_sha256=sha(dst), model_args=args,
                model_sha256=sha(ROOT/'model.py'), ignored_buffers=ignore,
                importer_sha256=sha(Path(__file__)), tensors=len(sd), dtype='float32')
    (ROOT/'host/manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
