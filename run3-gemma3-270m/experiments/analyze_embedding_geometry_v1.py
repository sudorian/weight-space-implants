"""Static exact token coordinates and native-attention projection costs.

The off coordinate is constrained to zero on digit/operator/BOS/newline
embeddings and tested on every other vocabulary embedding. No Gemma execution.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, atomic_torch, sha
from tools.tokenizer import Tokenizer

torch.set_num_threads(6)
torch.set_grad_enabled(False)
dst = ROOT / 'artifacts/embedding_geometry_v1.pt'
assert not dst.exists()
host = torch.load(ROOT / 'host/ckpt.pt', weights_only=True, mmap=True)['model']
tok = Tokenizer()
ids = [tok.encode(str(d))[0] for d in range(10)]+[tok.encode('+')[0], tok.encode('=')[0], 107, 2]
E = host['model.embed_tokens.weight']
scale = 640**.5
data = torch.load(ROOT / 'artifacts/host_calib256.pt', weights_only=True, mmap=True)
raw = data['captures']['0.raw'].reshape(-1, 640).double()
embedding = E[data['inputs'].flatten()].double()*scale
attention = raw-embedding
A = attention.T @ attention / len(attention)
C = torch.zeros(640, 640, dtype=torch.float64)
mu = torch.zeros(640, dtype=torch.float64)
for start in range(0, len(E), 4096):
    x = E[start:start+4096].double()*scale
    C += x.T @ x / len(E)
    mu += x.sum(0)/len(E)
A *= C.trace()/A.trace()
zero = (1+host['model.layers.0.pre_feedforward_layernorm.weight']) == 0
D = torch.cat((E[ids].double()*scale, torch.eye(640, dtype=torch.float64)[zero]))
_, singular, Vh = torch.linalg.svd(D, full_matrices=True)
rank = int((singular > singular.max()*1e-10).sum())
N = Vh[rank:].T
targets = torch.zeros(len(D), 3, dtype=torch.float64)
targets[:14, 0] = 1
targets[:10, 1] = torch.arange(10, dtype=torch.float64)
targets[10:14, 2] = torch.arange(1, 5, dtype=torch.float64)
mask = torch.ones(len(E), dtype=torch.bool)
mask[ids] = False
reports, results = [], {}
for mass in (0., .01, .1, 1., 10.):
    metric = C+mass*A
    metric.diagonal().add_(float(metric.trace()/640)*1e-7)
    off = N @ torch.linalg.solve(N.T @ metric @ N, N.T @ mu)
    off /= mu @ off
    B = torch.linalg.solve(metric, D.T)
    coordinates = B @ torch.linalg.solve(D @ B, targets)
    readouts = torch.cat((off[:, None], coordinates), 1)
    all_off = torch.empty(len(E), dtype=torch.float64)
    for start in range(0, len(E), 4096):
        all_off[start:start+4096] = E[start:start+4096].double() @ off * scale
    other = all_off[mask]
    report = {'attention_metric_mass': mass, 'constraint_rank': rank,
              'zero_pre_norm_coordinates': int(zero.sum()),
              'allowed_off_error': float((D @ off).abs().max()),
              'coordinate_error': float((D @ coordinates-targets).abs().max()),
              'other_off_min': float(other.min()), 'other_off_q001': float(torch.quantile(other, .001)),
              'other_off_q01': float(torch.quantile(other, .01)), 'nonpositive_other_count': int((other <= 0).sum()),
              'readout_max_abs': float(readouts.abs().max()),
              'native_attention_rms': (attention @ readouts).square().mean(0).sqrt().tolist()}
    reports.append(report)
    results[str(mass)] = {'readouts': readouts, 'report': report}
    print(json.dumps(report), flush=True)
atomic_torch(dst, {'results': results, 'reports': reports, 'allowed_token_ids': ids,
                  'fields': ['off', 'constant', 'digit', 'kind'], 'trial_masses': [0., .01, .1, 1., 10.],
                  'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}})
atomic_json('artifacts/embedding_geometry_v1.json', reports)
