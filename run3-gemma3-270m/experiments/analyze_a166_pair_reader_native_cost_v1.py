"""Two-source positional routing with zeros at both intended source positions.

Three rotary pairs form a symmetric two-peak kernel, zero at offsets +/-2
and negative at every other H1 position. A common positive key amplitude can
then vary without unbalancing the two intended source scores. Source labels
are positions only. Q/K matrices, RMS and RoPE are evaluated on recorded
states; no attention probabilities, values, GELU or complete model is run.
This is geometry only, not an addition checkpoint or a preservation claim.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
early, branch = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branch['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branch['inputs'])
def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
derived, identity = {}, {}
for tag in ['word', 'format']:
    state1 = early['captures'][f'{tag}.1.raw'] + norm(branch['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    state2 = state1 + norm(branch['captures'][f'{tag}.2.attention.raw']) * (1 + h['model.layers.2.post_attention_layernorm.weight'])
    identity[tag] = float((state2 - early['captures'][f'{tag}.2.raw']).abs().max())
    assert identity[tag] == 0
    derived[tag] = norm(state1) * (1 + h['model.layers.2.input_layernorm.weight'])
T = derived['word'].double()
SX = derived['format'][:, :9].double()
del early, branch, derived
train = T[:192].reshape(-1, 640); valid = T[192:]
P = SX[:, 8]; N = SX[:, :8].reshape(-1, 640)
pre = 'model.layers.2.self_attn.'
qn, kn, qgn, kgn = [pre + s for s in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight')]
Qh, Kh = h[qn].double().reshape(4, 256, 640), h[kn].double()
Gh, Hk = 1 + h[qgn].double(), 1 + h[kgn].double()
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * freq
    return x * angles.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * angles.sin().repeat(1, 2)
HQ = norm(torch.einsum('ntc,hjc->nthj', T, Qh)) * Gh
HK = norm(T @ Kh.T) * Hk
host_q = rotate(HQ[192:].permute(0, 2, 1, 3)); host_k = rotate(HK[192:])
host_scores = host_q @ host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); denom = causal.sum(-1)
def cost(Q, K, gq, gk):
    q = rotate((norm(torch.einsum('ntc,hjc->nthj', valid, Q)) * gq).permute(0, 2, 1, 3))
    k = rotate(norm(valid @ K.T) * gk)
    error = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (error * causal).sum(-1) / denom
    variance = ((error - mean[..., None]).square() * causal).sum(-1) / denom
    return {'centered_causal_score_mse': float(variance.mean()), 'p99_query_score_rms': float(variance.sqrt().flatten().quantile(.99))}
baseline = cost(ch[qn].double().reshape(4, 256, 640), ch.get(kn, h[kn]).double(),
                1 + ch[qgn].double(), 1 + ch.get(kgn, h[kgn]).double())

saved = torch.load(ROOT / 'artifacts/a165v1_zero_peak_pair_reader.pt', weights_only=True)
Q,K=saved['query'].double().reshape(4,256,640),saved['key'].double()
gq,gk=1+saved['query_norm'].double(),1+saved['key_norm'].double()
channels=torch.tensor(saved['report']['selected']['pairs']+[i+128 for i in saved['report']['selected']['pairs']])
Qzero,Kzero=Qh.clone(),Kh.clone();Qzero[:,channels]=0;Kzero[channels]=0
mean=P[:96].mean(0);qrow=saved['query_readout'].double()
Qcancel=Qh-torch.einsum('c,hjc->hj',mean,Qh)[:,:,None]*qrow
zeroq,zerok=gq.clone(),gk.clone();zeroq[channels]=0;zerok[channels]=0
cases={
 'current_reader':baseline,
 'selected_A165':cost(Q,K,gq,gk),
 'selected_A165_programme_gains_zero':cost(Q,K,zeroq,zerok),
 'only_six_native_channels_deleted':cost(Qzero,Kzero,Gh,Hk),
 'only_query_mean_cancel':cost(Qcancel,Kh,Gh,Hk),
 'only_query_changed':cost(Q,Kh,gq,Hk),
 'only_key_changed':cost(Qh,K,Gh,gk),
}
report={'analysis_only':True,'cases':cases,'selected_channels':channels.tolist(),
 'no_model_replay':True,'inputs':{'artifacts/a165v1_zero_peak_pair_reader.pt':sha(ROOT/'artifacts/a165v1_zero_peak_pair_reader.pt')},
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)}}
atomic_json(ROOT/'artifacts/a166v1_pair_reader_native_cost.json',report)
print(json.dumps(report,indent=2),flush=True)
