"""Five fixed native-energy-ranked single-pair deletion controls.

Only linear/RMS/RoPE algebra on recorded activations. No programme, attention
probabilities, new native nonlinearity, model, or weight checkpoint is made.
Deleting a pair is an optimistic diagnostic, not a mathematical lower bound.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
control_path = ROOT / 'artifacts/a273v1_sparse_query_energy_accommodation.json'
ch, ref = expand(h, base), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3:
        TH = (norm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + norm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
state = w['captures']['word.2.raw'] + norm(e['captures']['word.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
TW = (norm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
del state, w, e, host
pre = 'model.layers.3.self_attn.'
qn, qgn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
QH = h[qn].double().reshape(4, 256, 640); KH = h[kn].double()
gh, kh = 1 + h[qgn].double(), 1 + h[kgn].double()
frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

def raw_query(t, Q):
    return torch.stack([t @ Q[i].T for i in range(4)])

host_qn, host_kn = norm(raw_query(TH, QH)), norm(TH @ KH.T)
qenergy = (host_qn[:, :192] * gh).square().mean((0, 1, 2))
kenergy = (host_kn[:192] * kh).square().mean((0, 1))
qe, ke = qenergy[:128] + qenergy[128:], kenergy[:128] + kenergy[128:]
qf = host_qn[:, :192].square().mean((0, 1, 2))
kf = host_kn[:192].square().mean((0, 1))
qf, kf = (qf[:128] + qf[128:]) / 256, (kf[:128] + kf[128:]) / 256
importance = qe * ke
importance += importance.mean() * (qf / qf.mean() + kf / kf.mean())
gaps = 1 - (frequency[:, None] * torch.arange(1, 9, dtype=torch.float64)).cos().amax(1)
cost = importance.clone(); cost[gaps < .02] = torch.inf
pairs = cost.argsort()[:5]
assert torch.isfinite(cost[pairs]).all()
host_scores = rotate(host_qn[:, 192:] * gh) @ rotate(host_kn[192:] * kh).transpose(-1, -2)[None] / 16
same_qn, same_kn = norm(raw_query(TW, QH)), norm(TW @ KH.T)
same_scores = rotate(same_qn[:, 192:] * gh) @ rotate(same_kn[192:] * kh).transpose(-1, -2)[None] / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / counts
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / counts).mean())

def quality(Q, K, gq, gk):
    qq = norm(raw_query(TW[192:], Q)) * gq
    kk = norm(TW[192:] @ K.T) * gk
    scores = rotate(qq) @ rotate(kk).transpose(-1, -2)[None] / 16
    return {'paired_HOST_causal_score_MSE': centered_mse(scores - host_scores),
            'same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores),
            'paired_HOST_query_RMS': float((qq - host_qn[:, 192:] * gh).square().mean().sqrt()),
            'paired_HOST_key_RMS': float((kk - host_kn[192:] * kh).square().mean().sqrt())}

baseline = quality(ch[qn].double().reshape(4, 256, 640), KH, 1 + ch[qgn].double(), kh)
control = json.loads(control_path.read_text())['baseline']
assert abs(baseline['paired_HOST_causal_score_MSE'] - control['validation_paired_HOST_causal_score_MSE']) < 1e-12
assert abs(baseline['same_input_HOST_causal_score_MSE'] - control['validation_same_input_HOST_causal_score_MSE']) < 1e-12
native = quality(QH, KH, gh, kh)
print(json.dumps({'baseline': baseline, 'unmodified_native_weights_on_current_inputs': native,
                  'selected_pairs': pairs.tolist()}), flush=True)
trials = []
for pair in pairs.tolist():
    channels = torch.tensor([pair, pair + 128])
    Q, K = QH.clone(), KH.clone(); Q[:, channels] = 0; K[channels] = 0
    deletion = quality(Q, K, gh, kh)
    qq, kk = norm(raw_query(TW[:192], Q)), norm(TW[:192] @ K.T)
    gq = gh * (qq * same_qn[:, :192]).sum((0, 1, 2)) / qq.square().sum((0, 1, 2)).clamp_min(1e-30)
    gk = kh * (kk * same_kn[:192]).sum((0, 1)) / kk.square().sum((0, 1)).clamp_min(1e-30)
    gq[channels] = gh[channels]; gk[channels] = kh[channels]
    gq, gk = 1 + (gq - 1).float().double(), 1 + (gk - 1).float().double()
    accommodated = quality(Q, K, gq, gk)
    for metric in (deletion, accommodated):
        metric['promising'] = all(metric[key] <= 1.05 * baseline[key] for key in
                                  ('paired_HOST_causal_score_MSE', 'same_input_HOST_causal_score_MSE'))
    row = {'pair': pair, 'frequency': float(frequency[pair]), 'literal_minimum_cosine_gap': float(gaps[pair]),
           'native_energy_cost': float(importance[pair]), 'deletion': deletion, 'accommodated': accommodated,
           'maximum_query_gain_change': float((gq - gh).abs().max()),
           'maximum_key_gain_change': float((gk - kh).abs().max())}
    trials.append(row); print(json.dumps(row), flush=True)
report = {'analysis_only': True, 'baseline': baseline, 'unmodified_native_weights_on_current_inputs': native,
          'pair_ranking': 'original HOST normalized Q/K energy plus native raw-normalization fractions; one fixed ranking',
          'minimum_allowed_literal_cosine_gap': .02, 'valid_pairs': int((gaps >= .02).sum()),
          'trials': trials, 'any_promising': any(t[k]['promising'] for t in trials for k in ('deletion', 'accommodated')),
          'limitation': 'Programme-free deletion controls. Not a lower bound, complete reader or checkpoint. Adding programme may alter errors either way.',
          'training_windows': 192, 'holdback_windows': 64,
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, wp, ep, hp, control_path]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a283v1_single_pair_native_cost.json', report)
print(json.dumps({'any_promising': report['any_promising'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
