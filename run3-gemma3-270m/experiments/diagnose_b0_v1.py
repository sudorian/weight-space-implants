"""Projection diagnostics from matched, metered activations only."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json

torch.set_num_threads(6)
torch.set_grad_enabled(False)
sd = torch.load(ROOT / 'host/ckpt.pt', weights_only=True, mmap=True)['model']
c = torch.load(ROOT / 'experiments/b0_coordinates/circuit.pt', weights_only=True)
h = torch.load(ROOT / 'artifacts/host_detail_calib256.pt', weights_only=True, mmap=True)
b = torch.load(ROOT / 'artifacts/b0_calib256.pt', weights_only=True, mmap=True)
f = torch.load(ROOT / 'artifacts/b0_syntax_a.pt', weights_only=True, mmap=True)
assert torch.equal(h['inputs'], b['inputs'])
F, V = c['frame'], c['readouts']
R = torch.cat((V[:, :1], F), 1)
projector = R @ torch.linalg.solve(R.T @ R, R.T)
metric = torch.load(ROOT / 'artifacts/readout_v1.pt', weights_only=True)['vocabulary_metric']
report = {'text': {}, 'syntax': {}}
for tag in ('0.raw', '0.out', '1.out'):
    old = h['captures'][tag].reshape(-1, 640).double()
    new = b['captures'][tag].reshape(-1, 640).double()
    error = new-old
    mean = error.mean(0)
    fixed = error @ projector
    report['text'][tag] = {'rms': float(error.square().mean().sqrt()),
        'relative_squared_error': float(error.square().sum()/old.square().sum()),
        'mean_fraction': float(len(error)*mean.square().sum()/error.square().sum()),
        'protected_subspace_error_fraction': float(fixed.square().sum()/error.square().sum()),
        'weighted_rms': float(((error @ metric)*error).sum(1).mean().clamp_min(0).sqrt())}
ids = f['inputs'].flatten()
E = sd['model.embed_tokens.weight'][ids].double()*640**.5
pre = f['captures']['0.raw'].reshape(-1, 640).double()
out = f['captures']['0.out'].reshape(-1, 640).double()
report['syntax']['embedding_readout_error'] = ((pre-E) @ V).abs().max(0).values.tolist()
report['syntax']['post_mlp_off_error'] = float(((out-E) @ V[:, 0]).abs().max())
report['syntax']['post_mlp_frame_error'] = float(((out-E) @ F).abs().max())
report['syntax']['allowed_frame_native_max'] = float((out @ F).abs().max())
print(json.dumps(report, indent=2))
atomic_json('artifacts/b0_diagnosis.json', report)
