"""Prepare syntax-only probes; continuation digits are independent of operands."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.tokenizer import Tokenizer

tok = Tokenizer()
rng = np.random.default_rng(930173)
capture = {f'{l}.pre': {'module': f'model.layers.{l}.pre_feedforward_layernorm'} for l in range(18)}
capture.update({f'{l}.raw': {'module': f'model.layers.{l}.pre_feedforward_layernorm', 'kind': 'input'} for l in (0, 1, 2, 3, 16, 17)})
for name in ('syntax_a', 'syntax_b'):
    seqs = []
    for _ in range(512):
        aa, bb = map(int, rng.integers(100, 1000, 2))
        continuation = ''.join(map(str, rng.integers(0, 10, 5)))
        seqs.append(tok.encode(f'{aa}+{bb}={continuation}', bos=True))
    request = {'tokens': seqs, 'capture': capture, 'topk': 0, 'batch_size': 16}
    path = ROOT / f'experiments/host_{name}_request.json'
    assert not path.exists()
    path.write_text(json.dumps(request)+'\n')
seqs = []
roles = []
for i in range(512):
    aa, bb = map(int, rng.integers(100, 1000, 2))
    continuation = ''.join(map(str, rng.integers(0, 10, 5)))
    ids = tok.encode(f'{aa}+{bb}={continuation}', bos=True)
    role = i % 8
    replacement = tok.encode(['x', '-', '*', '/', '?', ' ', 'a', '.'][i//8 % 8])
    assert len(replacement) == 1
    ids[role+1] = replacement[0]
    roles.append(role)
    seqs.append(ids)
path = ROOT / 'experiments/host_malformed_request.json'
assert not path.exists()
path.write_text(json.dumps({'tokens': seqs, 'capture': capture, 'topk': 0, 'batch_size': 16})+'\n')
(ROOT / 'experiments/host_malformed_roles.json').write_text(json.dumps(roles)+'\n')
request = {'calib': {'count': 256, 'length': 64, 'seed': 41271}, 'capture': capture, 'topk': 0, 'batch_size': 8}
path = ROOT / 'experiments/host_calib256_request.json'
assert not path.exists()
path.write_text(json.dumps(request)+'\n')
print('Four syntax/calibration requests prepared; no addition targets used.')
