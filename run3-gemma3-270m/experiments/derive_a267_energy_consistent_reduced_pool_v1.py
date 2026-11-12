"""One-shot native repair for A266's reduced literal output geometry.

The original paired HOST state determines an exact raw-RMS-consistent target
on the old 96 strong rows. The complement remains fixed. Every fit uses this
same original target and a fixed algebraic gain-compensated reference map.
No candidate residuals, native activations or attention are recomputed.
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
desired = teacher[:, oldpool] - before[:, oldpool]

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
for count, geom in sorted(geometry['geometries'].items(), key=lambda z: int(z[0])):
    assert geom['report']['eligible_geometry'] and int(count) < 96
    pool = geom['pool'].long(); parameter = geom['gain_parameter'].float()
    gain = 1 + parameter.double()
    removed = oldpool[~torch.isin(oldpool, pool)]
    assert torch.equal(parameter[removed], h[pn][removed])
    assert torch.equal(parameter[pool], ch[pn][pool])
    info = {'pool_count': int(count), 'minimum_absolute_target_gain': float(gain[oldpool].abs().min())}
    if gain[oldpool].abs().min() <= 1e-9:
        info['eligible'] = False; info['reason'] = 'A target row has zero or nearly zero output gain'
        feasibility.append(info); continue
    y = desired / gain[oldpool]
    denominator = 640 - y.square().sum(1, keepdim=True)
    info.update(minimum_energy_denominator=float(denominator.min()),
        nonpositive_energy_denominators=int((denominator <= 0).sum()))
    if (denominator <= 0).any():
        info['eligible'] = False; info['reason'] = 'Original paired target has no real raw-RMS solution; no clipping or row deletion'
        feasibility.append(info); continue
    target_rms = ((complement_energy + 640e-6) / denominator).sqrt()
    target_rows = target_rms * y
    check_rms = ((complement_energy + target_rows.square().sum(1, keepdim=True)) / 640 + 1e-6).sqrt()
    info.update(eligible=True,
        maximum_relative_exact_energy_identity_error=float((check_rms / target_rms - 1).abs().max()),
        maximum_exact_target_branch_error=float((target_rows / check_rms * gain[oldpool] - desired).abs().max()),
        target_raw_RMS_ratio_minimum=float((target_rms / rms(raw0)).min()),
        target_raw_RMS_ratio_maximum=float((target_rms / rms(raw0)).max()))
    assert info['maximum_relative_exact_energy_identity_error'] < 1e-12
    assert info['maximum_exact_target_branch_error'] < 1e-9
    feasibility.append(info)
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
    'native_ridge_trials': ridges, 'correction_strength_trials': strengths, 'feasibility': feasibility,
    'trials': trials, 'full_vocabulary_validation': full, 'format_validation': fmt,
    'original_capture_reconstruction_error': recovery,
    'target': 'Original paired HOST final state on all original96 strong rows; exact raw RMS from fixed complement energy and new gains',
    'energy_equation': 'n^2=(sum(outside_raw^2)+640*epsilon)/(640-sum(((HOST-before)/new_gain)^2)); reject nonpositive denominator without clipping',
    'reference_map': 'Original A142 row map times original_gain/new_gain; algebraic prior, no residual fitting to a selected candidate',
    'selection': 'At least16 extra/8192 restricted and full-vocabulary matches; state/direction <=1.05baseline; word RMSchange<0.25; 2560/2560 formats with margin>1 and RMSchange<0.02; caps',
    'scope': 'Exact final linear/RMS arithmetic at recorded A142/A124 inputs. Installation on A259 is an upstream transfer requiring full score.',
    'no_model_execution_native_activation_replay_gradients_answers_or_iterative_residual_refits': True,
    'sources': {**geometry['report']['sources'], str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, install, reference, gp, wp, tp, hp, op, fp, frefp, base.parent / 'circuit.pt', install.with_suffix('.manifest.json'), ROOT / 'artifacts/a255v2_existing_anchor_readout_interval.json')},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a267v1_energy_consistent_reduced_pool.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'down': best[1], 'gain_parameter': best[2], 'geometry': best[4], 'report': report})
print(json.dumps({'baseline': baseline, 'feasibility': feasibility, 'selected': report['selected'],
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
