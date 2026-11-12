"""One-shot anchor repair from original HOST winner intervals on training text.

Each target is a minimum radial change needed for an original HOST winner,
subject to physical anchor-energy feasibility. Independent weighted linear
solves approximate those fixed targets. No candidate residual is refitted.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
install_base = ROOT / 'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
ch, ref = expand(h, base), expand(h, reference)
a, host, oldhost, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, op, tp)]
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert a['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(a['inputs'][256:], host['inputs'])
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units = c['units'].long()
F = a['captures']['17.features'].reshape(-1, 2048).double()
assert F[:, units].abs().max() == 0
before = a['captures']['17.raw'].reshape(-1, 640).double()
raw = F @ ch[dn].double().T
gain = 1 + ch[pn].double()
anchor = 439
assert abs(float(gain[anchor])) <= 1 and ch[dn][anchor, units].ne(0).any()

def rms(x):
    return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def norm(x):
    return x / rms(x)

rr = rms(raw)
original_branch = raw / rr * gain
parent = before + original_branch
branch = original_branch.clone(); branch[:, anchor] = 0
del original_branch
recorded = before + norm(F @ ref[dn].double().T) * (1 + ref[pn].double())
reconstruction = float((recorded - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded.norm())
assert reconstruction < 1e-6
del recorded, ref
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
installed = expand(h, install_base)
assert torch.equal(installed[dn], ch[dn]) and torch.equal(installed[pn], ch[pn])
del installed
nt = 384 * 64; nv = 128 * 64
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
ids = torch.cat([top['topk_ids'].reshape(-1, 8), a['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
tr = rms(teacher); parent_rms = rms(parent)
outside = (raw.square().sum(1) - raw[:, anchor].square()).clamp_min(0)
lower = torch.full((nt,), .5, dtype=torch.float64)
upper = torch.minimum(torch.full_like(lower, 2.), rr[:nt, 0] * (640 / (outside[:nt] + 640e-6)).sqrt())
assert (upper >= 1 - 1e-12).all()
margin_fractions = [0., .125, .25, .5, 1.]
mistake_weights = [1., 4., 16., 64., 256.]
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
target_factors = torch.ones(nt, len(margin_fractions), dtype=torch.float64)
training_wrong = torch.zeros(nt, dtype=torch.bool)
target_counts = [0] * len(margin_fractions)
for start in range(0, nt, 256):
    stop = min(start + 256, nt)
    vectors = E[ids[start:stop]].double() * fg
    winner_vectors = E[truth[start:stop]].double() * fg
    differences = winner_vectors[:, None] - vectors
    intercept = (differences * before[start:stop, None]).sum(-1)
    slope = (differences * branch[start:stop, None]).sum(-1)
    wrong = (intercept + slope < 0).any(1)
    training_wrong[start:stop] = wrong
    # Fixed original parent final RMS converts original HOST normalized
    # margins to affine raw-state inequalities. No candidate updates it.
    host_margin = ((differences * teacher[start:stop, None]).sum(-1)
        * (parent_rms[start:stop] / tr[start:stop])).clamp_min(0)
    for j, fraction in enumerate(margin_fractions):
        offset = intercept - fraction * host_margin
        crossing = -offset / torch.where(slope != 0, slope, torch.ones_like(slope))
        lo = torch.maximum(lower[start:stop], torch.where(slope > 0, crossing, -torch.inf).amax(1))
        hi = torch.minimum(upper[start:stop], torch.where(slope < 0, crossing, torch.inf).amin(1))
        feasible = (lo < hi) & ~((slope == 0) & (offset < 0)).any(1) & wrong
        wanted = torch.maximum(lo, torch.minimum(torch.ones_like(lo), hi))
        if fraction == 0:
            # One representable step into the interval, not a scale search.
            wanted = torch.where(wanted == lo, torch.nextafter(lo, hi), wanted)
            wanted = torch.where(wanted == hi, torch.nextafter(hi, lo), wanted)
        target_factors[start:stop, j] = torch.where(feasible, wanted, 1.)
        target_counts[j] += int(feasible.sum())
room = (640 * rr[:nt].square() / target_factors.square() - outside[:nt, None] - 640e-6).clamp_min(0)
sign = torch.where(raw[:nt, anchor] < 0, -1., 1.)
anchor_targets = room.sqrt() * sign[:, None]
Y = (anchor_targets - raw[:nt, anchor, None]) / rr[:nt]
# Preserve the original row target exactly on every unchanged observation.
Y[target_factors == 1] = 0
X = F[:nt, native] / rr[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15)
Z = X / sd
Fv = F[nt:, native].clone(); del F, X
row0 = ch[dn][anchor, native].double()
before_v, branch_v, teacher_v = before[nt:], branch[nt:], teacher[nt:]
rrv = rr[nt:, 0]; trv = tr[nt:, 0]
score_before = torch.empty(nv, 16, dtype=torch.float64)
score_branch = torch.empty_like(score_before)
baseline_count = 0
for start in range(0, nv, 256):
    stop = min(start + 256, nv)
    vectors = E[ids[nt+start:nt+stop]].double() * fg
    score_before[start:stop] = (vectors * before_v[start:stop, None]).sum(-1)
    score_branch[start:stop] = (vectors * branch_v[start:stop, None]).sum(-1)
    scores = (vectors * parent[nt+start:nt+stop, None]).sum(-1)
    prediction = ids[nt+start:nt+stop].gather(1, scores.argmax(1, keepdim=True)).flatten()
    baseline_count += int((prediction == truth[nt+start:nt+stop]).sum())
baseline = {'validation_restricted_matches': baseline_count,
    'validation_paired_HOST_state_relative_rms': float(((parent[nt:] - teacher_v) / tr[nt:]).square().mean().sqrt()),
    'validation_paired_HOST_direction_rms': float((norm(parent[nt:]) - norm(teacher_v)).square().mean().sqrt())}
control_path = ROOT / 'artifacts/a255v2_existing_anchor_readout_interval.json'
control = json.loads(control_path.read_text())
assert baseline_count == control['baseline']['validation_restricted_matches']
assert abs(baseline['validation_paired_HOST_state_relative_rms'] - control['baseline']['validation_paired_HOST_state_relative_rms']) < 1e-12
delta_before = before_v - teacher_v
e0 = delta_before.square().mean(1) / trv.square()
e1 = 2 * (delta_before * branch_v).mean(1) / trv.square()
e2 = branch_v.square().mean(1) / trv.square()
s0 = before_v.square().mean(1)
s1 = 2 * (before_v * branch_v).mean(1)
s2 = branch_v.square().mean(1)
t0 = (before_v * teacher_v).mean(1)
t1 = (branch_v * teacher_v).mean(1)
teacher_mean_square = teacher_v.square().mean(1)

def quality(row):
    anchor_value = Fv @ row.double()
    new_rms = ((outside[nt:] + anchor_value.square()) / 640 + 1e-6).sqrt()
    factor = rrv / new_rms
    scores = score_before + factor[:, None] * score_branch
    prediction = ids[nt:].gather(1, scores.argmax(1, keepdim=True)).flatten()
    state_mse = e0 + factor * e1 + factor.square() * e2
    mean_square = (s0 + factor * s1 + factor.square() * s2).clamp_min(0)
    direction_mse = (mean_square / (mean_square + 1e-6) + teacher_mean_square / trv.square()
        - 2 * (t0 + factor * t1) / (mean_square + 1e-6).sqrt() / trv)
    return {'validation_restricted_matches': int((prediction == truth[nt:]).sum()),
        'validation_paired_HOST_state_relative_rms': float(state_mse.clamp_min(0).mean().sqrt()),
        'validation_paired_HOST_direction_rms': float(direction_mse.clamp_min(0).mean().sqrt()),
        'maximum_validation_raw_RMS_relative_change': float((new_rms / rrv - 1).abs().max())}, factor

zero_gain_control, _ = quality(row0.float())
assert zero_gain_control['validation_restricted_matches'] == control['zero_gain_at_original_RMS']['validation_restricted_matches']
for key in ('validation_paired_HOST_state_relative_rms', 'validation_paired_HOST_direction_rms'):
    assert abs(zero_gain_control[key] - control['zero_gain_at_original_RMS'][key]) < 1e-10
bm = json.loads(install_base.with_suffix('.manifest.json').read_text())
gain_parameter = ch[pn].clone(); gain_parameter[anchor] = -1
gain_edit_change = int((gain_parameter.view(torch.int32) != h[pn].view(torch.int32)).sum()) - bm['per_tensor'][pn]
old_row_edits = int((ch[dn][anchor].view(torch.int32) != h[dn][anchor].view(torch.int32)).sum())
trials = []; best = None
for mistake_weight in mistake_weights:
    weights = torch.where(training_wrong, mistake_weight, 1.).double()
    weights /= weights.mean()
    covariance = Z.T @ (Z * weights[:, None]) / nt
    ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
    rhs = vec.T @ (Z.T @ (Y * weights[:, None]) / nt)
    for ridge in ridges:
        corrections = (vec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
        for j, fraction in enumerate(margin_fractions):
            for strength in strengths:
                row = (row0 + strength * corrections[:, j]).float()
                q, factor = quality(row)
                complete_row = ch[dn][anchor].clone(); complete_row[native] = row
                total = bm['changed_scalars'] + gain_edit_change - old_row_edits + int((complete_row.view(torch.int32) != h[dn][anchor].view(torch.int32)).sum())
                rec = {'mistake_weight': mistake_weight, 'HOST_margin_fraction': fraction,
                    'ridge': ridge, 'correction_strength': strength, **q,
                    'maximum_anchor_row_weight': float(complete_row.abs().max()),
                    'total_changed_scalars': total}
                rec['eligible'] = bool(row.isfinite().all() and complete_row.abs().max() <= 1e6 and total <= 5016268
                    and q['validation_restricted_matches'] >= baseline_count + 16
                    and q['validation_paired_HOST_state_relative_rms'] <= 1.05 * baseline['validation_paired_HOST_state_relative_rms']
                    and q['validation_paired_HOST_direction_rms'] <= 1.05 * baseline['validation_paired_HOST_direction_rms']
                    and q['maximum_validation_raw_RMS_relative_change'] < .25)
                trials.append(rec)
                key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
                if rec['eligible'] and (best is None or key > best[0]):
                    best = key, complete_row.clone(), factor.clone(), rec
    print(json.dumps({'mistake_weight': mistake_weight, 'selected': None if best is None else best[-1]}), flush=True)
del Z, covariance, ev, vec, rhs, corrections

def full_readout(state, expected):
    x = (norm(state) * fg).float()
    maximum = torch.full((len(x),), -torch.inf)
    winner = torch.full_like(expected, -1); competitor = torch.full_like(maximum, -torch.inf)
    wanted = (x * E[expected]).sum(1)
    for start in range(0, len(E), 2048):
        logits = x @ E[start:start+2048].T
        values, indices = logits.max(1); better = values > maximum
        maximum[better] = values[better]; winner[better] = start + indices[better]
        local = expected-start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        competitor = torch.maximum(competitor, logits.amax(1))
    return {'tokens': len(x), 'matches': int((winner == expected).sum()),
        'minimum_target_margin': float((wanted-competitor).min())}

provisional = None; full = None; fmt = None
format_path = ROOT / 'artifacts/a124v1_syntax_stress.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
if best is not None:
    provisional = best[-1]
    candidate_state = before_v + best[2][:, None] * branch_v
    full = {'parent': full_readout(parent[nt:], truth[nt:]),
        'candidate': full_readout(candidate_state, truth[nt:])}
    if full['candidate']['matches'] < full['parent']['matches'] + 16:
        best = None
if best is not None:
    stress = torch.load(format_path, weights_only=True, mmap=True)
    fref = expand(h, format_reference)
    assert stress['checkpoint_sha256'] == sha(format_reference)
    for name in h:
        if name not in (dn, pn):
            assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
    del fref
    SF = stress['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
    SB = stress['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
    old_raw = SF @ ch[dn].double().T
    new_raw = old_raw.clone(); new_raw[:, anchor] = SF @ best[1].double()
    new_gain = 1 + gain_parameter.double()
    expected = stress['topk_ids'][:, :, 0].reshape(-1).long()
    fmt = {'parent': full_readout(SB + norm(old_raw) * gain, expected),
        'candidate': full_readout(SB + norm(new_raw) * new_gain, expected),
        'maximum_raw_RMS_relative_change': float((rms(new_raw) / rms(old_raw) - 1).abs().max())}
    assert fmt['parent']['matches'] == len(expected)
    if fmt['candidate']['matches'] != len(expected) or fmt['candidate']['minimum_target_margin'] <= 1 or fmt['maximum_raw_RMS_relative_change'] >= .02:
        best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'install_base_checkpoint': str(install_base.relative_to(ROOT)),
    'anchor': anchor, 'original_anchor_output_gain': float(gain[anchor]), 'proposed_anchor_output_gain': 0.,
    'training_windows': 384, 'validation_windows': 128, 'training_zero_gain_parent_mistakes': int(training_wrong.sum()),
    'margin_fraction_trials': margin_fractions, 'mistake_weight_trials': mistake_weights,
    'ridge_trials': ridges, 'correction_strength_trials': strengths,
    'training_repairable_mistakes_per_margin_fraction': target_counts,
    'target_relative_radial_domain': [.5, 2.], 'original_anchor_sign_retained_in_targets': True,
    'target': 'Minimum feasible radial movement to obtain a fixed fraction of original HOST winner/rival margins, only on original zero-gain parent training mistakes; unchanged observations and infeasible cases retain the original anchor target.',
    'target_normalizers': 'Original parent raw and final RMS and original HOST final RMS, fixed before all independent solves.',
    'selection_guard': 'At least16extra restricted then full-vocabulary matches/8192; state/direction <=1.05baseline; maximum word raw-RMS change<0.25; all2560format winners with margin>1 and raw-RMS change<0.02; caps.',
    'baseline': baseline, 'zero_gain_control': zero_gain_control, 'trials': trials,
    'provisional_selection': provisional, 'full_vocabulary_validation': full, 'format_validation': fmt,
    'selected': None if best is None else best[-1],
    'transfer_caveat': 'A142 has exact captured inputs. Intended A240 changes earlier readers; identical final Down/gains are checked, but its complete trajectory is not locally observed.',
    'programme_columns_G_Up_and_every_other_Down_row_unchanged': True,
    'no_gradients_native_activation_replay_arithmetic_answers_or_iterative_residual_refitting': True,
    'sources': {**control['sources'], str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {**control['inputs'], **{str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent/'circuit.pt', install_base, reference, ap, hp, op, tp, control_path, format_path, format_reference)}},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a258v1_winner_interval_anchor.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'anchor_row': best[1], 'gain_parameter': gain_parameter, 'report': report})
print(json.dumps({'selected': report['selected'], 'full_vocabulary_validation': full,
    'format_validation': fmt, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
