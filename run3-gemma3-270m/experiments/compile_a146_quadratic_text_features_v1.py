"""One-shot original-HOST readout repair with paired quadratic text features.

On recorded word states, gates k*g0 +/- u are strictly positive and share
Up=v/2. Their difference is the hand-built polynomial u*v. On independent
output-format states both gates are strictly negative. No native activation
function or Gemma computation is replayed. A fixed input PCA, native/product
Schur complement and rank-revealing QR select products from unlabeled text.
Each allocation then uses independent original-target closed-form solves.
"""
import copy
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha, ELIGIBLE

h = host_state()
base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
hp = ROOT / 'artifacts/a56v1_host.pt'
np = ROOT / 'artifacts/a140v1_host_branches.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
ch = expand(h, base)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
w, top, oldhost, newhost, fmt = [torch.load(p, weights_only=True, mmap=True) for p in (wp, tp, hp, np, fp)]
assert w['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'], top['inputs'])
assert torch.equal(w['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(w['inputs'][256:], newhost['inputs'])
assert w['logit_positions'] == list(range(64))
for key, value in w['captures'].items():
    if key.endswith('program_features'):
        assert value.abs().max() == 0, key
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
W0 = ch[dn].double(); gamma = 1 + ch[pn].double()
units = c['units'].long(); pool = c['output_pool'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert len(pool) == 96 and len(units) == 11 and (gamma[pool] == 1e6).all()
F = w['captures']['17.features'].double().reshape(-1, 2048)
raw = w['captures']['17.down'].double().reshape(-1, 640)
before = w['captures']['17.raw'].double().reshape(-1, 640)
current = w['captures']['finalraw'].double().reshape(-1, 640)
teacher = torch.cat([oldhost['captures']['finalraw'][:256], newhost['captures']['finalraw']], 0).double().reshape(-1, 640)
nt = 384 * 64
def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
cr = (current.square().mean(1, keepdim=True) + 1e-6).sqrt()
assert (F @ W0.T - raw).norm() / raw.norm() < 1e-5
assert (before + norm(raw) * gamma - current).norm() / current.norm() < 1e-5
teacher_n, current_n = norm(teacher), norm(current)
del teacher
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
host_ids = top['topk_ids'].reshape(-1, 8).long()
edited_ids = w['topk_ids'].reshape(-1, 8).long()
truth = host_ids[nt:, 0]
gap = (top['topk_values'].reshape(-1, 8)[:, 0] - top['topk_values'].reshape(-1, 8)[:, 1]).double().clamp_min(0)
weight = 1 / (gap + .1); weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
sqrt_weight = weight.sqrt()
readout_d = (E[host_ids[:, 0]].double() - E[host_ids[:, 1]].double()) * fg
metric = readout_d[:nt].T @ (readout_d[:nt] * weight[:nt, None]) / nt
metric = (metric + metric.T) / 2
metric_scale = float(metric.diagonal().mean())
small_metric = metric[pool][:, pool].clone(); small_metric.diagonal().add_(.01 * metric_scale)
projection = torch.linalg.solve(small_metric, metric[pool]).T
metric_factor = torch.linalg.cholesky(small_metric)
pg = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
Xpre = norm(before) * pg
PF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
PB = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
Ppre = norm(PB) * pg
format_reference = norm(fmt['captures']['finalraw'][:, 8:13].double().reshape(-1, 640))
format_margin = (fmt['topk_values'][:, :, 0] - fmt['topk_values'][:, :, 1]).double().reshape(-1)
maximum_readout_norm = 0.
for start in range(0, len(E), 4096):
    maximum_readout_norm = max(maximum_readout_norm, float((E[start:start + 4096].double() * fg).norm(dim=1).max()))
validation_ids = torch.cat([host_ids[nt:], edited_ids[nt:]], 1)
rivals = E[validation_ids].double() * fg[None, None]

# The inverse existing arithmetic gate supplies a word/format sign, never an
# arithmetic answer. Conservative dot-rounding allowance is checked below.
g0 = -ch[gn][units[0]].double()
wg, ag = Xpre @ g0, Ppre @ g0
round_factor = 650 * torch.finfo(torch.float32).eps / (1 - 650 * torch.finfo(torch.float32).eps)
wg_error = round_factor * (Xpre.abs() @ g0.abs())
ag_error = round_factor * (Ppre.abs() @ g0.abs())
assert (wg - wg_error).min() > 0 and (-ag - ag_error).min() > 0

# Fixed 64-PC directions; no target-dependent input direction fit.
mean = Xpre[:nt].mean(0)
cov = Xpre[:nt].T @ Xpre[:nt] / nt - torch.outer(mean, mean)
ev, V = torch.linalg.eigh((cov + cov.T) / 2)
directions = V[:, -64:]
P = Xpre @ directions
direction_scale = P[:nt].square().mean(0).sqrt().clamp_min(1e-12)
directions /= direction_scale
P = Xpre @ directions
pairs = torch.triu_indices(64, 64).T
assert len(pairs) == 2080
Q = P[:, pairs[:, 0]] * P[:, pairs[:, 1]]
selection_factor = sqrt_weight[:nt, None] / rms[:nt] / cr[:nt]
XN = F[:nt, native] * selection_factor
nsd = XN.square().mean(0).sqrt().clamp_min(1e-15); XN /= nsd
QQ = Q[:nt] * selection_factor
qsd = QQ.square().mean(0).sqrt().clamp_min(1e-15); QQ /= qsd
target_selection = ((teacher_n[:nt] - current_n[:nt]) @ projection @ metric_factor) * sqrt_weight[:nt, None]
CN = XN.T @ XN / nt; CN.diagonal().add_(.001)
cross = XN.T @ QQ / nt
solved_cross = torch.linalg.solve(CN, cross)
rhs = XN.T @ target_selection / nt
conditional_cross = QQ.T @ target_selection / nt - cross.T @ torch.linalg.solve(CN, rhs)
conditional_variance = (QQ.square().mean(0) - (cross * solved_cross).sum(0)).clamp_min(1e-12)
qr_rows = conditional_cross / conditional_variance.sqrt()[:, None]
selection_order = []; qr_report = []
allowed = torch.ones(len(pairs), dtype=torch.bool)
for _ in range(32):
    energy = qr_rows.square().sum(1); energy[~allowed] = -torch.inf
    j = int(energy.argmax()); assert energy[j] > 1e-20
    selection_order.append(j); allowed[j] = False
    direction = qr_rows[j] / qr_rows[j].norm()
    qr_report.append({'product_index': j, 'input_pc_pair': pairs[j].tolist(),
                      'conditional_variance': float(conditional_variance[j]), 'qr_pivot_energy': float(energy[j])})
    qr_rows -= (qr_rows @ direction)[:, None] * direction[None]
del Q, QQ, XN, CN, cross, solved_cross, rhs, conditional_cross, qr_rows, P, target_selection

# Quantized paired gates retain signed margins after the conservative dot
# allowance. For |gate|>=32, tanh-GELU differs from its linear/zero ideal by
# less than 1e-100; no tanh, GELU or native activation evaluation occurs here.
gates = []; ups = []; quadratic = []; gate_report = []
for j in selection_order:
    i, k = pairs[j].tolist(); u, v = directions[:, i], directions[:, k]
    wu, au = Xpre @ u, Ppre @ u
    wu_error = round_factor * (Xpre.abs() @ u.abs())
    au_error = round_factor * (Ppre.abs() @ u.abs())
    scale = max(float(((64.1 + wu.abs() + wu_error) / (wg - wg_error)).max()),
                float(((64.1 + au.abs() + au_error) / (-ag - ag_error)).max())) * 1.001
    G = torch.stack([scale * g0 + u, scale * g0 - u]).float()
    U = (v / 2).float()
    gw, ga = Xpre @ G.double().T, Ppre @ G.double().T
    ew = round_factor * (Xpre.abs() @ G.double().abs().T)
    ea = round_factor * (Ppre.abs() @ G.double().abs().T)
    signed_minimum = min(float((gw - ew).min()), float((-ga - ea).min()))
    assert signed_minimum > 32 and max(float(G.abs().max()), float(U.abs().max())) <= 1e6
    q = (gw[:, 0] - gw[:, 1]) * (Xpre @ U.double())
    quadratic.append(q); gates.append(G); ups.append(U)
    gate_report.append({'input_pc_pair': [i, k], 'common_gate_scale': scale,
                        'minimum_signed_gate_after_dot_allowance': signed_minimum,
                        'maximum_gate_weight': float(G.abs().max()), 'maximum_Up_weight': float(U.abs().max())})
Q = torch.stack(quadratic, 1); del quadratic
Gpairs = torch.stack(gates); Upairs = torch.stack(ups)
all_donors = c['native_donor_repairs']['17']['donors'].long()
assert len(all_donors) == 512 and not torch.isin(all_donors, units).any()
feature_energy = (F[:nt, all_donors] / rms[:nt]).square().mean(0)
output_energy = (W0[:, all_donors] * gamma[:, None]).square().sum(0)
quiet_order = all_donors[(feature_energy * output_energy).argsort()]
counts = [0, 4, 8, 16, 32]
ridges = [.001, .01, .1, 1., 10.]
strengths = [.25, .5, .75, 1., 1.25]
out = ROOT / 'experiments/a146v1_quadratic_text_features/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
trials = []; best_zero = None; best_positive = None

def quality(new_raw, new_format_raw):
    nrms = (new_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    state = before[nt:] + new_raw / nrms * gamma
    scores = torch.einsum('nd,nkd->nk', state, rivals)
    winners = validation_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
    difference = norm(state) - teacher_n[nt:]
    boundary = (difference * readout_d[nt:]).sum(1)
    final = norm(PB + norm(new_format_raw) * gamma)
    bound = float((format_margin - 2 * (final - format_reference).norm(dim=1) * maximum_readout_norm).min())
    return {'validation_restricted_matches': int((winners == truth).sum()),
            'validation_boundary_score_rms': float((boundary.square() * weight[nt:]).sum().sqrt() / weight[nt:].sum().sqrt()),
            'validation_final_direction_rms': float(difference.square().mean().sqrt()),
            'maximum_relative_raw_norm_change': float((nrms / rms[nt:] - 1).abs().max()),
            'minimum_recorded_format_margin_bound': bound}, state

baseline, base_state = quality(raw[nt:], PF @ W0.T)
print(json.dumps({'baseline': baseline, 'gate_minimum': min(x['minimum_signed_gate_after_dot_allowance'] for x in gate_report),
                  'selected_products': qr_report}), flush=True)
for count in counts:
    removed = quiet_order[:2 * count]
    donors = all_donors[~torch.isin(all_donors, removed)]
    surviving = native[~torch.isin(native, removed)]
    Wfixed = W0.clone()
    if count:
        D = F[:nt, donors] / rms[:nt]
        sd = D.square().mean(0).sqrt().clamp_min(1e-15); Z = D / sd
        C = Z.T @ Z / nt; C.diagonal().add_(.001)
        removed_fit = torch.linalg.solve(C, Z.T @ (F[:nt, removed] / rms[:nt]) / nt) / sd[:, None]
        Wfixed[:, donors] += W0[:, removed] @ removed_fit.T
        Wfixed[:, removed] = 0
        del D, Z, C, removed_fit
    Wfixed = Wfixed.float().double()
    fixed_raw = F @ Wfixed.T
    fixed_format_raw = PF @ Wfixed.T
    linear_state = before / cr + fixed_raw / rms / cr * gamma
    target = (teacher_n - linear_state) @ projection
    features = torch.cat([F[:, surviving], Q[:, :count]], 1)
    X = features[:nt] / rms[:nt] / cr[:nt] * sqrt_weight[:nt, None]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
    C = Z.T @ Z / nt
    rhs = Z.T @ (target[:nt] * sqrt_weight[:nt, None]) / nt
    ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    projected = V.T @ rhs
    initial = torch.cat([Wfixed[pool[:, None], surviving[None]], torch.zeros(len(pool), count, dtype=torch.float64)], 1)
    del X, Z, C, rhs, linear_state, target
    for ridge in ridges:
        beta = ((V @ (projected / (ev[:, None] + ridge))) / sd[:, None]).T / gamma[pool, None]
        for strength in strengths:
            rows = (initial + strength * beta).float()
            delta = rows.double() - initial
            new_raw = fixed_raw[nt:].clone(); new_raw[:, pool] += features[nt:] @ delta.T
            new_format_raw = fixed_format_raw.clone()
            new_format_raw[:, pool] += PF[:, surviving] @ delta[:, :len(surviving)].T
            q, states = quality(new_raw, new_format_raw)
            row = {'quadratic_product_count': count, 'new_unit_count': 2 * count,
                   'native_ridge': ridge, 'correction_strength': strength, **q,
                   'maximum_new_output_weight': float(rows.abs().max())}
            row['eligible'] = bool(rows.isfinite().all() and rows.abs().max() <= 1e6 and
                q['maximum_relative_raw_norm_change'] < .02 and q['minimum_recorded_format_margin_bound'] > 1. and
                q['validation_final_direction_rms'] < 1.2 * baseline['validation_final_direction_rms'])
            trials.append(row); print(json.dumps(row), flush=True)
            key = (q['validation_restricted_matches'], -q['validation_boundary_score_rms'])
            best = best_zero if count == 0 else best_positive
            if row['eligible'] and (best is None or key > best[0]):
                W = Wfixed.float().clone()
                W[pool[:, None], surviving[None]] = rows[:, :len(surviving)]
                if count:
                    W[pool[:, None], removed[0::2][None]] = rows[:, len(surviving):]
                    W[pool[:, None], removed[1::2][None]] = -rows[:, len(surviving):]
                changed = bm['changed_scalars'] - int((ch[dn].view(torch.int32) != h[dn].view(torch.int32)).sum()) + int((W.view(torch.int32) != h[dn].view(torch.int32)).sum()) + 2 * 640 * len(removed)
                row['changed_scalars_upper_bound'] = changed
                if changed <= int(.05 * ELIGIBLE):
                    entry = key, W, states.clone(), row, removed.clone()
                    if count == 0: best_zero = entry
                    else: best_positive = entry
    atomic_json(out.parent / 'partial_trials.json', {'baseline': baseline, 'trials': trials})
    del features, fixed_raw, fixed_format_raw, V, projected, beta

report = {'baseline': baseline, 'quadratic_product_count_trials': counts,
          'new_unit_count_trials': [0, 8, 16, 32, 64], 'input_pca_rank': 64, 'candidate_product_count': 2080,
          'feature_selection_native_ridge': .001, 'feature_selection': 'One fixed native/product Schur complement and rank-revealing QR of target cross-covariance',
          'selected_product_order': qr_report, 'paired_gate_checks': gate_report,
          'dot_rounding_factor': round_factor, 'GELU_error_bound_per_unit': 'Less than 1e-100 times abs(Up projection) when signed gate remains at least32',
          'new_gate_rule': 'k times inverse existing common final arithmetic gate, plus/minus one PCA direction; common Up half another PCA direction',
          'native_displacement_donor_ridge': .001, 'native_ridge_trials': ridges, 'correction_strength_trials': strengths,
          'training_windows': 384, 'fresh_validation_windows': 128, 'margin_floor': .1, 'maximum_token_weight': 100.,
          'boundary_metric_ridge_fraction': .01, 'fixed_normalizers': 'Recorded original A123 raw MLP RMS and final-state RMS',
          'original_arithmetic_features_Down_gains_and_upstream_fixed': True,
          'no_answers_gradients_activation_replay_or_iterative_refit': True,
          'trials': trials, 'best_zero': None if best_zero is None else best_zero[3],
          'selected': None if best_positive is None else best_positive[3],
          'maximum_resident_set_kib_before_readout': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(out.parent / 'quadratic_text_repair.json', report)
assert best_zero is not None and best_positive is not None, 'No eligible control or quadratic repair'

def matches(states):
    x = (norm(states) * fg).float()
    maxima = torch.full((len(x),), -torch.inf); winners = torch.full((len(x),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = x @ E[start:start + 4096].T
        values, indices = logits.max(1); better = values > maxima
        maxima[better] = values[better]; winners[better] = start + indices[better]
    return int((winners == truth).sum())
full = {'tokens': len(truth), 'baseline_matches': matches(base_state),
        'zero_quadratic_control_matches': matches(best_zero[2]), 'quadratic_matches': matches(best_positive[2]),
        'required_extra_matches_over_control': 8}
report['full_vocabulary_validation'] = full
atomic_json(out.parent / 'quadratic_text_repair.json', report)
print(json.dumps({'full_vocabulary_validation': full}), flush=True)
assert full['quadratic_matches'] >= full['zero_quadratic_control_matches'] + 8, 'Less than eight extra fresh matches over the zero-quadratic control'

_, W, _, selected, removed = best_positive
count = selected['quadratic_product_count']
G, U = ch[gn].clone(), ch[un].clone()
G[removed] = Gpairs[:count].reshape(-1, 640)
U[removed] = Upairs[:count, None].expand(-1, 2, -1).reshape(-1, 640)
assert torch.equal(W[:, units], ch[dn][:, units])
assert torch.equal(G[units], ch[gn][units]) and torch.equal(U[units], ch[un][units])
ch[gn], ch[un], ch[dn] = G, U, W
c['quadratic_text_features_a146'] = report
c['quadratic_text_features_a146']['repair_units'] = removed.tolist()
atomic_torch(out.parent / 'circuit.pt', c)
categories = copy.deepcopy(bm['categories'])
for name in (gn, un):
    mask = ch[name].view(torch.int32) != h[name].view(torch.int32)
    categories[name] = {'arithmetic_features': int(mask[units].sum()), 'quadratic_text_features': int(mask[removed].sum())}
changed = W.view(torch.int32) != h[dn].view(torch.int32)
categories[dn] = {'arithmetic_programme': int(changed[:, units].sum()),
                  'native_and_quadratic_text_repair': int(changed[:, native].sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a142_paired_host_readout_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, tp, hp, np, fp],
     arguments={}, description=__doc__, extra=report, categories=categories)
