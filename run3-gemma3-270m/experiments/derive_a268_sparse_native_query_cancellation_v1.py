"""Reconstruct A-reader queries with sparse ordinary-row mean cancellation.

Keep the original A61 programmed-channel native map and position direction.
Ordinary rows start at HOST, with one sparse format-mean cancellation. Fit
ordinary gains once to original paired HOST queries, then derive routing
amplitudes analytically. The goal is to free at least300k edits with bounded
causal-score error. No probabilities, native activations or answers are used.
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
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
original = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
control_path = ROOT / 'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
historical_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch, old, ref = expand(h, base), expand(h, original), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), old.get(name, h[name]))
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
c = torch.load(original.parent / 'circuit.pt', weights_only=True)
selected = torch.tensor(c['native_a_stage']['reader']['native_key_query_layer3']['selected_channels'])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), selected)]
assert len(selected) == 64 and len(ordinary) == 192
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3: TH = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
inputs = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
TW = inputs['word']; T = TW.reshape(-1, 640); current = inputs['format'][:, :9]
historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in historical_paths]).double()
SX = torch.cat([current, historical]); nt = 192 * 64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (kn, kgn): assert torch.equal(ch.get(name, h[name]), h[name])
assert torch.equal(ch[gn], old[gn])
QH = h[qn].double().reshape(4, 256, 640)
Q0 = old[qn].double().reshape(4, 256, 640)
Qleader = ch[qn].double().reshape(4, 256, 640)
gh = 1 + h[gn].double(); g0 = 1 + old[gn].double()
K = h[kn].double(); kg = 1 + h[kgn].double()
quiet = torch.load(qp, weights_only=True)['query'].double()
old_mean = historical[:1024, 8].mean(0)
qnull = quiet / (old_mean @ quiet)
B = QH - (old_mean @ QH.transpose(-1, -2))[:, :, None] * qnull
native_program_rows = B[:, selected] * (gh[selected] / g0[selected])[None, :, None]
P = torch.zeros_like(Q0); P[:, selected] = Q0[:, selected] - native_program_rows
amplitude = (P @ quiet) / quiet.square().sum()
decomposition_error = float((P - amplitude[:, :, None] * quiet).abs().max())
assert decomposition_error < 1e-6
frequency = 10000.**(-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

rotated_keys = rotate(norm(TW @ K.T) * kg)
host_keys = rotate(norm(TH @ K.T) * kg)
host_queries = torch.stack([norm(TH @ QH[i].T) * gh for i in range(4)])
host_scores = rotate(host_queries) @ host_keys.transpose(-1, -2)[None] / 16
same_scores = torch.stack([rotate(norm(TW @ QH[i].T) * gh) @ rotated_keys.transpose(-1, -2) / 16 for i in range(4)])
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
causal = torch.ones(64, 64, dtype=torch.float64).tril(); count_causal = causal.sum(-1)
positions = [3, 2, 1, 0]

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / count_causal
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / count_causal).mean())

def quality(query, gain):
    post = torch.stack([norm(TW[192:] @ query[i].T) * gain for i in range(4)])
    scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2)[None] / 16
    gaps = []
    for head, pos in enumerate(positions):
        q = norm(SX[:, 8] @ query[head].T) * gain
        ss = (KR * q[:, None]).sum(-1) / 16
        others = torch.tensor([i for i in range(9) if i != pos])
        gaps.append(ss[:, pos] - ss[:, others].amax(1))
    gaps = torch.stack(gaps)
    return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[:, 192:]),
        'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[:, 192:]),
        'validation_paired_HOST_query_RMS': float((post - host_queries[:, 192:]).square().mean().sqrt()),
        'minimum_source_routing_margin': float(gaps.min()),
        'current32_source_margin': float(gaps[:, 96:128].min()), 'historical512_source_margin': float(gaps[:, -512:].min())}

baseline = quality(Qleader, g0)
original_quality = quality(Q0, g0)
control = json.loads(control_path.read_text())['combined_candidate']
assert abs(baseline['validation_paired_HOST_causal_score_MSE'] - control['A237_paired_HOST_score_MSE']) < 1e-12
assert abs(original_quality['validation_paired_HOST_causal_score_MSE'] - control['baseline_centered_causal_score_mse']) < 1e-12
# One fixed sparse-support ranking, based only on original text covariance
# and current format means/variation. No candidate prediction enters it.
mu = current[:96, 8].mean(0); centered = current[:96, 8] - mu
Cword = T[:nt].T @ T[:nt] / nt
Cf = centered.T @ centered / 96
metric = Cword + 10000 * Cf
metric.diagonal().add_(1e-6 * Cword.diagonal().mean())
precision = torch.linalg.inv(metric)
full = precision @ mu
order = (full.square() / precision.diagonal().clamp_min(1e-30)).argsort(descending=True)
supports = [1, 4, 16, 64, 128]
programme_source = torch.einsum('nc,hjc->nhj', SX[:, 8], P)

def balance(native, gain):
    nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
    factors = []; margins = []
    for head, pos in enumerate(positions):
        others = torch.tensor([i for i in range(9) if i != pos])
        difference = KR[:, pos, None] - KR[:, others]
        D = (difference * (nr[:, head] * gain)[:, None]).sum(-1) / 16
        E = (difference * (programme_source[:, head] * gain)[:, None]).sum(-1) / 16
        aa = programme_source[:, head].square().mean(1, keepdim=True)
        bb = 2 * (nr[:, head] * programme_source[:, head]).mean(1, keepdim=True)
        cc = nr[:, head].square().mean(1, keepdim=True) + 1e-6
        A = E.square() - 64 * aa; Bq = 2 * D * E - 64 * bb; Cq = D.square() - 64 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Program-only asymptotic routing does not exceed8'}
        discriminant = Bq.square() - 4 * A * Cq
        root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        gap = (D + factor * E) / (cc + factor * bb + factor**2 * aa).sqrt()
        assert gap.min() >= 7.999
        factors.append(factor); margins.append(float(gap.min()))
    return torch.tensor(factors, dtype=torch.float64), {'analytic_programme_factors': factors, 'analytic_minimum_source_margins': margins}

bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials = []; best = None
for count in supports:
    columns = order[:count]
    q = torch.zeros(640, dtype=torch.float64)
    q[columns] = torch.linalg.solve(metric[columns][:, columns], mu[columns])
    q /= mu @ q
    N = QH.clone(); N[:, selected] = native_program_rows
    means = torch.einsum('c,hjc->hj', mu, QH[:, ordinary])
    N[:, ordinary] -= means[:, :, None] * q
    # Fit shared ordinary gains once on original HOST-at-HOST-input targets.
    # The original programme amplitude is used here and never refitted after
    # the subsequent exact source-routing calculation.
    fit_query = N + P
    numerator = torch.zeros(256, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for head in range(4):
        x = norm(T[:nt] @ fit_query[head].T)
        target = host_queries[head, :192].reshape(nt, 256)
        numerator += (x * target).sum(0)
        denominator += x.square().sum(0)
    gain = g0.clone(); gain[ordinary] = numerator[ordinary] / denominator[ordinary].clamp_min(1e-30)
    gain_parameter = (gain - 1).float(); gain = 1 + gain_parameter.double()
    factors, routing = balance(N, gain)
    rec = {'ordinary_cancellation_support': count, 'input_columns': columns.tolist(),
        'minimum_current_format_cancellation_response': float((current[:, 8] @ q).min()),
        'maximum_current_format_cancellation_response': float((current[:, 8] @ q).max()),
        'maximum_cancellation_readout_weight': float(q.abs().max()), **routing}
    if factors is None:
        rec['eligible'] = False; trials.append(rec); print(json.dumps(rec), flush=True); continue
    query = (N + factors[:, None, None] * P).float()
    outside = torch.arange(640)[~torch.isin(torch.arange(640), columns)]
    assert torch.equal(query[:, ordinary[:, None], outside[None]], h[qn].reshape(4, 256, 640)[:, ordinary[:, None], outside[None]])
    assert torch.equal(gain_parameter[selected], ch[gn][selected])
    qscore = quality(query.double(), gain)
    total = bm['changed_scalars'] - bm['per_tensor'][qn] - bm['per_tensor'][gn]
    total += int((query.reshape(1024, 640).view(torch.int32) != h[qn].view(torch.int32)).sum())
    total += int((gain_parameter.view(torch.int32) != h[gn].view(torch.int32)).sum())
    rec.update(qscore, total_changed_scalars=total, freed_scalars=bm['changed_scalars'] - total,
        maximum_query_weight=float(query.abs().max()), maximum_query_norm_weight=float(gain_parameter.abs().max()))
    rec['eligible'] = bool(query.isfinite().all() and gain_parameter.isfinite().all()
        and max(rec['maximum_query_weight'], rec['maximum_query_norm_weight']) <= 1e6
        and total <= 5016268 and rec['freed_scalars'] >= 300000
        and qscore['minimum_source_routing_margin'] >= 7.999
        and qscore['validation_paired_HOST_causal_score_MSE'] <= 1.05 * baseline['validation_paired_HOST_causal_score_MSE']
        and qscore['validation_same_input_HOST_causal_score_MSE'] <= 1.05 * baseline['validation_same_input_HOST_causal_score_MSE']
        and qscore['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS'])
    trials.append(rec)
    key = (qscore['validation_paired_HOST_causal_score_MSE'], -rec['freed_scalars'])
    if rec['eligible'] and (best is None or key < best[0]):
        best = key, query.reshape(1024, 640).clone(), gain_parameter.clone(), rec
    print(json.dumps(rec), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'original_query_checkpoint': str(original.relative_to(ROOT)), 'baseline': baseline,
    'original_query_quality': original_quality, 'selected': None if best is None else best[-1],
    'training_word_windows': 192, 'validation_word_windows': 64, 'format_mean_windows': 96,
    'all_source_routing_windows': len(SX), 'support_count_trials': supports,
    'fixed_format_variation_mass': 10000., 'fixed_covariance_ridge_fraction': 1e-6,
    'programmed_channels': selected.tolist(), 'programme_rank_one_decomposition_error': decomposition_error,
    'trials': trials,
    'construction': 'HOST ordinary query rows minus sparse format mean; original A61 selected native rows and programme direction; one original paired-HOST shared ordinary-gain solve; exact final quadratic routing amplitude',
    'selection': 'Free>=300000edits; paired/same-input causal QK MSE and paired query RMS<=1.05leader; allsourcegaps>=7.999; caps',
    'sparse_support': 'All ordinary-channel columns outside the selected support remain bit-identical HOST; selected64channels remain dense',
    'routing_scope': 'Amplitude bounds use every1664known source window; current32/historical512 margins are diagnostics, not unseen routing tests',
    'scope': 'No attention probabilities or downstream trajectory is recomputed; freeing edits is not itself a text-preservation claim',
    'no_native_activation_execution_gradients_whole_answers_or_iterative_candidate_refits': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp, control_path, original.parent / 'circuit.pt', base.with_suffix('.manifest.json')] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a268v1_sparse_native_query_cancellation.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'query_gain_parameter': best[2], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
