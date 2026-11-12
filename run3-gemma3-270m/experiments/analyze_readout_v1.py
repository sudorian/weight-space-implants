"""Weight-only decimal and newline output geometry; no model execution.

Three-vector parabolic output is derived from the fixed unembedding. Only five
class-bias scales are tested, on scalar digit values, never addition problems.
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
destination = ROOT / 'artifacts/readout_v1.pt'
assert not destination.exists()
sd = torch.load(ROOT / 'host/ckpt.pt', weights_only=True, mmap=True)['model']
tok = Tokenizer()
ids = [tok.encode(str(d))[0] for d in range(10)]+[107]
E = sd['model.embed_tokens.weight']
gamma = 1+sd['model.norm.weight'].double()
C = torch.zeros(640, 640, dtype=torch.float64)
mu = torch.zeros(640, dtype=torch.float64)
for start in range(0, len(E), 4096):
    A = E[start:start+4096].double()*gamma
    C += A.T @ A / len(E)
    mu += A.sum(0)/len(E)
C -= mu[:, None]*mu[None, :]
C.diagonal().add_(1e-5*C.trace()/640)
D = E[ids].double()*gamma
B = torch.linalg.solve(C, D.T)
W = torch.linalg.solve(D @ B, B.T).T
assert (D @ W-torch.eye(11, dtype=torch.float64)).abs().max() < 1e-7
d = torch.arange(10, dtype=torch.float64)-4.5
reports, vectors = [], {}
for bias in (16., 32., 64., 128., 256.):
    base = W[:, :10] @ (bias-d.square())
    linear = W[:, :10] @ (2*d)
    normalizer = max(float((base+linear*(x-4.5)).norm()) for x in (0., 9.))
    base, linear = base/normalizer, linear/normalizer
    values = torch.tensor([v+error for v in range(10) for error in (-.2, 0., .2)], dtype=torch.float64)
    frame = base[:, None]+linear[:, None]*(values-4.5)
    frame /= frame.norm(dim=0)
    best = torch.full((len(values),), -float('inf'), dtype=torch.float64)
    best_ids = torch.zeros(len(values), dtype=torch.long)
    correct_logits = ((D[torch.arange(10).repeat_interleave(3)]*frame.T).sum(1))*640**.5
    other = torch.full_like(best, -float('inf'))
    for start in range(0, len(E), 4096):
        z = (E[start:start+4096].double()*gamma) @ frame * 640**.5
        vals, which = z.max(0)
        replace = vals > best
        best[replace] = vals[replace]
        best_ids[replace] = which[replace]+start
        for j, desired in enumerate(torch.tensor(ids[:10]).repeat_interleave(3).tolist()):
            if start <= desired < start+len(z):
                z[desired-start, j] = -float('inf')
        other = torch.maximum(other, z.max(0).values)
    report = {'class_bias': bias, 'minimum_margin': float((correct_logits-other).min()),
              'integer_margins': (correct_logits-other)[1::3].tolist(),
              'top_ids': best_ids.tolist(), 'normalizer': normalizer}
    reports.append(report)
    vectors[str(bias)] = {'constant': base, 'linear': linear, 'newline': W[:, 10]/W[:, 10].norm()}
    print(json.dumps(report), flush=True)
chosen = max(reports, key=lambda r: r['minimum_margin'])
atomic_torch(destination, {'vectors': vectors, 'reports': reports, 'chosen_bias': chosen['class_bias'],
                          'ridge_fraction': 1e-5, 'class_bias_trials': [16., 32., 64., 128., 256.],
                          'token_ids': ids, 'onehot_readout': W, 'vocabulary_metric': C,
                          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}})
atomic_json('artifacts/readout_v1.json', reports)
