"""Decompose recorded-text QK error into native-pair loss and programme terms.

These algebraic counterfactual score matrices are diagnostics, not runnable
checkpoints. There is no softmax, activation evaluation or model execution.
"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state()
base = ROOT / 'experiments/a98v1_pooled_gate_bounds/delta.pt'; ch = expand(h, base)
wp = ROOT / 'artifacts/a72v2_word.pt'; rp = ROOT / 'artifacts/a43_all_head_queries_v1.pt'
pair = int(torch.load(rp, weights_only=True)['report']['pair'])
X = torch.load(wp, weights_only=True, mmap=True)['captures']['7.input'][384:].double()
pre = 'model.layers.7.self_attn.'
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def normal(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def rotate(x):
    a = torch.arange(64, dtype=torch.float64)[:, None] * freq
    return x * a.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * a.sin().repeat(1, 2)
def maps(state):
    Q = state[pre+'q_proj.weight'].double().reshape(4, 256, 640)
    K = state[pre+'k_proj.weight'].double()
    qr = torch.einsum('ntc,hjc->nthj', X, Q).permute(0, 2, 1, 3)
    kr = X @ K.T
    q = rotate(normal(qr) * (1 + state[pre+'q_norm.weight'].double()))
    k = rotate(normal(kr) * (1 + state[pre+'k_norm.weight'].double()))
    return q, k, qr.square().mean(-1)+1e-6, kr.square().mean(-1)+1e-6
eq, ek, eqe, eke = maps(ch); hq, hk, hqe, hke = maps(h)
channels = [pair, pair+128]
hp = hq[..., channels] @ hk[:, None, :, channels].transpose(-1, -2) / 16
ep = eq[..., channels] @ ek[:, None, :, channels].transpose(-1, -2) / 16
err = eq @ ek[:, None].transpose(-1, -2) / 16 - hq @ hk[:, None].transpose(-1, -2) / 16
ordinary = err - ep + hp
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
def centered(x): return (x - ((x*causal).sum(-1)/counts)[...,None]) * causal
def mse(x): return float((centered(x).square().sum(-1)/counts).mean())
terms = {'ordinary_error': ordinary, 'programme_pair': ep, 'removed_HOST_pair': hp}
cross = {a: {b: float(((centered(x)*centered(y)).sum(-1)/counts).mean())
             for b,y in terms.items()} for a,x in terms.items()}
report = {'pair': pair, 'baseline_mse': mse(err), 'component_mse': {k:mse(v) for k,v in terms.items()},
          'cross_moments': cross, 'ideal_native_pair_restored_mse': mse(ordinary+ep),
          'programme_pair_absent_mse': mse(ordinary-hp), 'ordinary_exact_mse': mse(ep-hp),
          'query_relative_energy_rms': float((eqe/hqe-1).square().mean().sqrt()),
          'key_relative_energy_rms': float((eke/hke-1).square().mean().sqrt()),
          'not_runnable_counterfactuals': True, 'source_sha256':sha(Path(__file__)),
          'inputs': {str(p.relative_to(ROOT)):sha(p) for p in (base,wp,rp)}}
atomic_json(ROOT/'artifacts/a107v1_copy_pair_error.json',report)
print(json.dumps(report,indent=2),flush=True)
