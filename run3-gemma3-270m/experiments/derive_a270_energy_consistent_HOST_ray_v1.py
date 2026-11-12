"""One-shot native repair from exact positive HOST-ray targets.

Allow a positive final-state scale alpha. On the fixed complement, its
least-squares value is affine in inverse raw RMS. Substitution into the raw
energy identity gives one quadratic; examine its two roots directly. This
is original-target algebra, not iterative fitting of a candidate residual.
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
install = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
gp = ROOT / 'artifacts/a266v1_competitor_penalized_output_pool.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
frefp = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ch, ref = expand(h, base), expand(h, reference)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
geometry, word, top, host, oldhost = [torch.load(p, weights_only=True, mmap=True)
    for p in (gp, wp, tp, hp, op)]
for p, digest in geometry['report']['sources'].items(): assert sha(ROOT / p) == digest
for p, digest in geometry['report']['inputs'].items(): assert sha(ROOT / p) == digest
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
gn = 'model.layers.17.mlp.gate_proj.weight'
un = 'model.layers.17.mlp.up_proj.weight'
for name in h:
    if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert word['checkpoint_sha256'] == sha(reference)
assert torch.equal(word['inputs'], top['inputs'])
assert torch.equal(word['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(word['inputs'][256:], host['inputs'])
installed = expand(h, install)
for name in (gn, un, dn, pn): assert torch.equal(ch[name], installed[name]), name
del installed
units = c['units'].long(); oldpool = c['output_pool'].long()
assert len(units) == 11 and len(oldpool) == 96
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
outside = torch.arange(640)[~torch.isin(torch.arange(640), oldpool)]
F = word['captures']['17.features'].reshape(-1, 2048).double()
assert F[:, units].abs().max() == 0
before = word['captures']['17.raw'].reshape(-1, 640).double()
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
W0 = ch[dn].double(); gain0 = 1 + ch[pn].double()
raw0 = F @ W0.T

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)

recorded = before + norm(F @ ref[dn].double().T) * (1 + ref[pn].double())
recovery = float((recorded - word['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded.norm())
assert recovery < 1e-6
del recorded, ref
parent = before + norm(raw0) * gain0
tr = rms(teacher)
nt = 384 * 64; nv = 128 * 64
F = F[:, native].contiguous()
ids = torch.cat([top['topk_ids'].reshape(-1, 8), word['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
complement_energy = raw0[:, outside].square().sum(1, keepdim=True)
teacher_complement = teacher[:, outside]
ray_denominator = teacher_complement.square().sum(1, keepdim=True)
assert ray_denominator.min() > 0
ray_intercept = (teacher_complement * before[:, outside]).sum(1, keepdim=True) / ray_denominator
ray_slope = (teacher_complement * (raw0[:, outside] * gain0[outside])).sum(1, keepdim=True) / ray_denominator
geometries = dict(geometry['geometries'])
geometries['96'] = {'pool': oldpool, 'basis': c['output_directions'],
    'programme': ch[dn][:, units].clone(), 'gain_parameter': ch[pn].clone(),
    'report': {'eligible_geometry': True, 'original96_control': True,
        'pool_count': 96, 'anchor_amplitude': c['piece']['raw_anchor_amplitude'],
        'raw_output_gain': c['piece']['output_gain'], 'analytic_cleanup_scale_ratio': 1.}}
del teacher_complement

def quality(states):
    matches = 0
    for start in range(0, nv, 256):
        stop = min(start + 256, nv)
        rivals = E[ids[nt + start:nt + stop]].double() * fg
        scores = (states[start:stop, None] * rivals).sum(-1)
        winners = ids[nt + start:nt + stop].gather(1, scores.argmax(1, keepdim=True)).flatten()
        matches += int((winners == truth[nt + start:nt + stop]).sum())
    return {'validation_restricted_matches': matches,
        'validation_paired_HOST_state_relative_rms': float(((states - teacher[nt:]) / tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(states) - norm(teacher[nt:])).square().mean().sqrt())}

baseline = quality(parent[nt:])
control = json.loads((ROOT / 'artifacts/a255v2_existing_anchor_readout_interval.json').read_text())
for k in baseline: assert abs(baseline[k] - control['baseline'][k]) < 1e-10, k
bm = json.loads(install.with_suffix('.manifest.json').read_text())
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
trials = []; feasibility = []; best = None
for count, geom in sorted(geometries.items(), key=lambda z: int(z[0])):
    assert geom['report']['eligible_geometry'] and int(count) <= 96
    pool = geom['pool'].long(); parameter = geom['gain_parameter'].float()
    gain = 1 + parameter.double()
    removed = oldpool[~torch.isin(oldpool, pool)]
    assert torch.equal(parameter[removed], h[pn][removed])
    assert torch.equal(parameter[pool], ch[pn][pool])
    info = {'pool_count': int(count), 'minimum_absolute_target_gain': float(gain[oldpool].abs().min())}
    if gain[oldpool].abs().min() <= 1e-9:
        info['eligible'] = False; info['reason'] = 'A target row has zero or nearly zero output gain'
        feasibility.append(info); continue
    u = (ray_intercept * teacher[:, oldpool] - before[:, oldpool]) / gain[oldpool]
    v = ray_slope * teacher[:, oldpool] / gain[oldpool]
    aa = v.square().sum(1, keepdim=True) + complement_energy + 640e-6
    bb = 2 * (u * v).sum(1, keepdim=True)
    cc = u.square().sum(1, keepdim=True) - 640
    discriminant = bb.square() - 4 * aa * cc
    # Stable quadratic roots; negative-discriminant rows remain infeasible.
    qq = -.5 * (bb + torch.where(bb >= 0, 1., -1.) * discriminant.clamp_min(0).sqrt())
    roots = torch.cat([qq / aa, cc / torch.where(qq != 0, qq, torch.nan)], 1)
    alphas = ray_intercept + ray_slope * roots
    valid = (discriminant >= 0) & roots.isfinite() & (roots > 1e-12) & alphas.isfinite() & (alphas > 1e-12)
    costs = torch.full_like(roots, torch.inf)
    for index in range(2):
        t = torch.where(valid[:, index:index+1], roots[:, index:index+1], torch.ones_like(ray_intercept))
        alpha = torch.where(valid[:, index:index+1], alphas[:, index:index+1], torch.ones_like(ray_intercept))
        state = before + raw0 * t * gain
        state[:, oldpool] = alpha * teacher[:, oldpool]
        error = (norm(state) - norm(teacher)).square().mean(1)
        costs[:, index] = torch.where(valid[:, index], error, torch.inf)
    feasible = valid.any(1)
    info.update(negative_quadratic_discriminants=int((discriminant < 0).sum()),
        tokens_without_positive_ray_root=int((~feasible).sum()))
    if not feasible.all():
        info['eligible'] = False; info['reason'] = 'Original ray target has no positive RMS/scale root; no clipping or row deletion'
        feasibility.append(info); continue
    choice = costs.argmin(1, keepdim=True)
    t = roots.gather(1, choice); alpha = alphas.gather(1, choice)
    desired = alpha * teacher[:, oldpool] - before[:, oldpool]
    y = desired / gain[oldpool]
    target_rms = 1 / t
    target_rows = target_rms * y
    check_rms = ((complement_energy + target_rows.square().sum(1, keepdim=True)) / 640 + 1e-6).sqrt()
    ideal_state = before + raw0 / check_rms * gain
    ideal_state[:, oldpool] = before[:, oldpool] + target_rows / check_rms * gain[oldpool]
    orthogonal = (teacher[:, outside] * (ideal_state[:, outside] - alpha * teacher[:, outside])).sum(1, keepdim=True) / ray_denominator
    info.update(eligible=True, minimum_positive_HOST_scale=float(alpha.min()), maximum_positive_HOST_scale=float(alpha.max()),
        root_choice_counts=[int((choice == i).sum()) for i in range(2)],
        maximum_complement_ray_normal_equation_error=float(orthogonal.abs().max()),
        maximum_relative_exact_energy_identity_error=float((check_rms / target_rms - 1).abs().max()),
        maximum_exact_target_branch_error=float((target_rows / check_rms * gain[oldpool] - desired).abs().max()),
        target_raw_RMS_ratio_minimum=float((target_rms / rms(raw0)).min()),
        target_raw_RMS_ratio_maximum=float((target_rms / rms(raw0)).max()),
        ideal_target_validation_without_fitted_weights=quality(ideal_state[nt:]))
    assert info['maximum_complement_ray_normal_equation_error'] < 1e-10
    assert info['maximum_relative_exact_energy_identity_error'] < 1e-10
    assert info['maximum_exact_target_branch_error'] < 1e-7
    feasibility.append(info)
    del ideal_state, roots, alphas, costs, valid, u, v
    # This reference is a direct algebraic transfer of the original map,
    # not a fitted candidate. It defines the fixed ridge prior only.
    Wref = W0[oldpool[:, None], native[None]] * (gain0[oldpool] / gain[oldpool])[:, None]
    X = F[:nt] / target_rms[:nt] / tr[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15)
    Z = X / sd
    target = (desired[:nt] - (F[:nt] @ Wref.T) / target_rms[:nt] * gain[oldpool]) / tr[:nt]
    covariance = Z.T @ Z / nt
    ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
    rhs = vec.T @ (Z.T @ target / nt)
    del X, Z, covariance, target
    for ridge in ridges:
        correction = ((vec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]).T / gain[oldpool, None]
        for strength in strengths:
            rows = (Wref + strength * correction).float()
            W = ch[dn].clone(); W[oldpool[:, None], native[None]] = rows
            W[:, units] = geom['programme']
            raw = raw0[nt:].clone(); raw[:, oldpool] = F[nt:] @ rows.double().T
            state = before[nt:] + norm(raw) * gain
            q = quality(state)
            total = bm['changed_scalars'] - bm['per_tensor'][dn] - bm['per_tensor'][pn]
            total += int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            total += int((parameter.view(torch.int32) != h[pn].view(torch.int32)).sum())
            rec = {'pool_count': int(count), 'native_ridge': ridge, 'correction_strength': strength,
                **q, 'maximum_validation_raw_RMS_relative_change': float((rms(raw) / rms(raw0[nt:]) - 1).abs().max()),
                'maximum_native_row_weight': float(rows.abs().max()), 'total_changed_scalars': total}
            rec['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6 and total <= 5016268
                and q['validation_restricted_matches'] >= baseline['validation_restricted_matches'] + 16
                and q['validation_paired_HOST_state_relative_rms'] <= 1.05 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] <= 1.05 * baseline['validation_paired_HOST_direction_rms']
                and rec['maximum_validation_raw_RMS_relative_change'] < .25)
            trials.append(rec)
            key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
            if rec['eligible'] and (best is None or key > best[0]):
                best = key, W.clone(), parameter.clone(), state.clone(), geom, rec
    del ev, vec, rhs, correction
    print(json.dumps({'feasibility': info, 'selected': None if best is None else best[-1]}), flush=True)

def full_readout(states, expected):
    x = (norm(states) * fg).float()
    maximum = torch.full((len(x),), -torch.inf); winners = torch.full_like(expected, -1)
    other = torch.full_like(maximum, -torch.inf); wanted = (x * E[expected]).sum(1)
    for start in range(0, len(E), 2048):
        logits = x @ E[start:start + 2048].T
        values, indices = logits.max(1); take = values > maximum
        maximum[take] = values[take]; winners[take] = start + indices[take]
        local = expected - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, logits.amax(1))
    return {'tokens': len(x), 'matches': int((winners == expected).sum()),
        'minimum_target_margin': float((wanted - other).min())}

provisional = None; full = None; fmt = None
if best is not None:
    provisional = best[-1]
    full = {'parent': full_readout(parent[nt:], truth[nt:]), 'candidate': full_readout(best[3], truth[nt:])}
    if full['candidate']['matches'] < full['parent']['matches'] + 16: best = None
if best is not None:
    f = torch.load(fp, weights_only=True, mmap=True); fref = expand(h, frefp)
    assert f['checkpoint_sha256'] == sha(frefp)
    for name in h:
        if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
    del fref
    SF = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
    SB = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
    old_raw = SF @ W0.T; new_raw = SF @ best[1].double().T
    expected = f['topk_ids'][:, :, 0].reshape(-1).long()
    fmt = {'parent': full_readout(SB + norm(old_raw) * gain0, expected),
        'candidate': full_readout(SB + norm(new_raw) * (1 + best[2].double()), expected),
        'maximum_raw_RMS_relative_change': float((rms(new_raw) / rms(old_raw) - 1).abs().max())}
    assert fmt['parent']['matches'] == 2560
    if fmt['candidate']['matches'] != 2560 or fmt['candidate']['minimum_target_margin'] <= 1 or fmt['maximum_raw_RMS_relative_change'] >= .02:
        best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'install_base_checkpoint': str(install.relative_to(ROOT)), 'baseline': baseline,
    'selected': None if best is None else best[-1], 'provisional_before_full_checks': provisional,
    'training_windows': 384, 'validation_windows': 128, 'native_feature_count': len(native),
    'geometry_pool_counts': sorted(int(k) for k in geometries), 'native_ridge_trials': ridges, 'correction_strength_trials': strengths, 'feasibility': feasibility,
    'trials': trials, 'full_vocabulary_validation': full, 'format_validation': fmt,
    'original_capture_reconstruction_error': recovery,
    'target': 'Positive ray through original paired HOST final state on original96 rows; complement least-squares ray scale affine in inverse raw RMS, coupled to exact raw energy by a quadratic',
    'energy_equation': 'alpha=a+b*t; sum(((alpha*HOST_S-before_S)/gain_S)^2)+t^2*(outside_raw_energy+640*epsilon)=640; t=1/raw_RMS; inspect both exact quadratic roots, require t>0 and alpha>0, choose minimum original HOST direction error',
    'reference_map': 'Original A142 row map times original_gain/new_gain; algebraic prior, no residual fitting to a selected candidate',
    'selection': 'At least16 extra/8192 restricted and full-vocabulary matches; state/direction <=1.05baseline; word RMSchange<0.25; 2560/2560 formats with margin>1 and RMSchange<0.02; caps',
    'scope': 'Exact final linear/RMS arithmetic at recorded A142/A124 inputs. Installation on A259 is an upstream transfer requiring full score.',
    'no_model_execution_native_activation_replay_gradients_answers_or_iterative_residual_refits': True,
    'sources': {**geometry['report']['sources'], str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a267_energy_consistent_reduced_pool_v1.py': sha(ROOT/'experiments/derive_a267_energy_consistent_reduced_pool_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, install, reference, gp, wp, tp, hp, op, fp, frefp, base.parent / 'circuit.pt', install.with_suffix('.manifest.json'), ROOT / 'artifacts/a255v2_existing_anchor_readout_interval.json')},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a270v1_energy_consistent_HOST_ray.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'down': best[1], 'gain_parameter': best[2], 'geometry': best[4], 'report': report})
print(json.dumps({'baseline': baseline, 'feasibility': feasibility, 'selected': report['selected'],
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
