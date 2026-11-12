"""One original-target native-feature fit of joint output/anchor geometry.

The A287 training targets are fixed before every independent solve. Original
HOST Down supplies the prior. Only recorded features and linear/RMS/readout
algebra are used; there is no activation replay or iterative residual fit.
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
gp = ROOT / 'artifacts/a287v1_joint_projected_anchor_geometry.pt'
geo = torch.load(gp, weights_only=True)
for category in ('sources', 'inputs'):
    for path, digest in geo['report'][category].items():
        assert sha(ROOT / path) == digest, path
assert geo['report']['any_promising_geometry']
base = ROOT / geo['report']['comparison_checkpoint']
reference = ROOT / geo['report']['reference_checkpoint']
install_base = ROOT / geo['report']['possible_installation_base']
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
op = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ch, ref, installed = [expand(h, p) for p in (base, reference, install_base)]
a, host, oldhost, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, op, tp)]
dn, pn = 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert a['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256])
assert torch.equal(a['inputs'][256:], host['inputs'])
assert torch.equal(ch[dn], installed[dn]) and torch.equal(ch[pn], installed[pn])
for name in ('model.layers.17.mlp.gate_proj.weight', 'model.layers.17.mlp.up_proj.weight'):
    assert torch.equal(ch[name], installed[name])
del installed
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units, pool = c['units'].long(), c['output_pool'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
rows = geo['rows'].long(); anchor = 439
assert torch.equal(rows[:-1], pool) and rows[-1] == anchor
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), rows)]
assert torch.equal(ch[dn][ordinary], ref[dn][ordinary])
assert torch.equal(ch[dn][:, units], ref[dn][:, units])
F = a['captures']['17.features'].reshape(-1, 2048)
assert F[:, units].abs().max() == 0
before = a['captures']['17.raw'].reshape(-1, 640).double()
raw = F.double() @ ref[dn].double().T
old_gain = 1 + ch[pn].double()
gain_parameter = ch[pn].clone(); gain_parameter[anchor] = -1
gain = 1 + gain_parameter.double()
assert torch.equal(gain, geo['gain'])

def rms(x):
    return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def norm(x):
    return x / rms(x)

rr = rms(raw)
observed = before + raw / rr * old_gain
reconstruction = float((observed - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / observed.norm())
assert reconstruction < 1e-6
comparison_raw = F.double() @ ch[dn].double().T
comparison = before + norm(comparison_raw) * old_gain
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
nt, nv = 384 * 64, 128 * 64
ids = torch.cat([top['topk_ids'].reshape(-1, 8), a['topk_ids'].reshape(-1, 8)], 1).long()
truth = top['topk_ids'].reshape(-1, 8)[:, 0].long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
gap = (top['topk_values'].reshape(-1, 8)[:, 0] - top['topk_values'].reshape(-1, 8)[:, 1]).double().clamp_min(0)
weight = 1 / (gap + .1); weight /= weight[:nt].mean(); weight = weight.clamp(max=100)
prior = h[dn][rows[:, None], native[None]].double().clone()
prior[:-1] *= ((1 + h[pn].double())[pool] / gain[pool])[:, None]
prior = prior.float().double()
X = F[:nt, native].double() / rr[:nt]
Xw = X * weight[:nt, None].sqrt()
sd = Xw.square().mean(0).sqrt().clamp_min(1e-15)
Z = Xw / sd
cov = Z.T @ Z / nt
ev, vec = torch.linalg.eigh((cov + cov.T) / 2); ev = ev.clamp_min(0)
fractions = [0., .125, .25, .5, 1.]
assert set(geo['training_raw_targets_over_reference_RMS']) == {str(f) for f in fractions}
original_prior_predictions = X @ prior.T
rhs_parts = []
for fraction in fractions:
    target = geo['training_raw_targets_over_reference_RMS'][str(fraction)]
    assert target.shape == (nt, 97)
    rhs_parts.append(Z.T @ ((target - original_prior_predictions) * weight[:nt, None].sqrt()) / nt)
rhs = vec.T @ torch.cat(rhs_parts, 1)
Fv = F[nt:, native].double()
reference_rows = ref[dn][rows[:, None], native[None]].double()
del X, Xw, Z, cov, rhs_parts, original_prior_predictions, target, a, host, oldhost, top, geo
before_v, raw_v, teacher_v = before[nt:], raw[nt:], teacher[nt:]
tr = rms(teacher_v); teacher_n = norm(teacher_v)
fixed_scores = torch.empty(nv, 16, dtype=torch.float64)
before_scores = torch.empty_like(fixed_scores)
row_readouts = torch.empty(nv, 16, 97, dtype=torch.float64)
comparison_matches = 0
for start in range(0, nv, 256):
    end = min(start + 256, nv)
    vectors = E[ids[nt + start:nt + end]].double() * fg
    before_scores[start:end] = (vectors * before_v[start:end, None]).sum(-1)
    fixed_scores[start:end] = (vectors * (raw_v[start:end] * gain)[..., None, :]).sum(-1)
    row_readouts[start:end] = vectors[:, :, rows] * gain[rows]
    scores = (vectors * comparison[nt + start:nt + end, None]).sum(-1)
    prediction = ids[nt + start:nt + end].gather(1, scores.argmax(1, keepdim=True)).flatten()
    comparison_matches += int((prediction == truth[nt + start:nt + end]).sum())
baseline = {'validation_restricted_matches': comparison_matches,
            'validation_HOST_state_relative_RMS': float(((comparison[nt:] - teacher_v) / tr).square().mean().sqrt()),
            'validation_HOST_direction_RMS': float((norm(comparison[nt:]) - teacher_n).square().mean().sqrt())}
assert comparison_matches == 8003
bm = json.loads(install_base.with_suffix('.manifest.json').read_text())
replaced_old_count = int((ch[dn][rows[:, None], native[None]].contiguous().view(torch.int32)
                         != h[dn][rows[:, None], native[None]].contiguous().view(torch.int32)).sum())
gain_delta = int((gain_parameter.view(torch.int32) != h[pn].view(torch.int32)).sum()) - bm['per_tensor'][pn]
HOST_rows = h[dn][rows[:, None], native[None]].contiguous().view(torch.int32)

def quality(candidate):
    change = Fv @ (candidate.double() - reference_rows).T
    nr = raw_v.clone(); nr[:, rows] += change
    nrms = rms(nr)
    scores = before_scores + (fixed_scores + (row_readouts * change[:, None]).sum(-1)) / nrms
    prediction = ids[nt:].gather(1, scores.argmax(1, keepdim=True)).flatten()
    state = before_v + nr / nrms * gain
    return {'validation_restricted_matches': int((prediction == truth[nt:]).sum()),
            'validation_HOST_state_relative_RMS': float(((state - teacher_v) / tr).square().mean().sqrt()),
            'validation_HOST_direction_RMS': float((norm(state) - teacher_n).square().mean().sqrt()),
            'maximum_validation_raw_RMS_relative_change': float((nrms / rr[nt:] - 1).abs().max())}, state

ridges = [1e-6, 1e-5, 1e-4, .001, .01]
strengths = [.25, .5, .75, 1., 1.25]
trials = []; best = None; diagnostic = None
print(json.dumps({'baseline': baseline, 'original_state_reconstruction_relative_error': reconstruction}), flush=True)
for ridge in ridges:
    all_corrections = ((vec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]).T.reshape(5, 97, len(native))
    for j, fraction in enumerate(fractions):
        for strength in strengths:
            candidate = (prior + strength * all_corrections[j]).float()
            q, state = quality(candidate)
            total = bm['changed_scalars'] - replaced_old_count + gain_delta
            total += int((candidate.contiguous().view(torch.int32) != HOST_rows).sum())
            record = {'ridge': ridge, 'HOST_margin_fraction': fraction, 'strength': strength,
                      **q, 'total_changed_scalars': total, 'maximum_fitted_native_weight': float(candidate.abs().max())}
            record['eligible'] = bool(q['validation_restricted_matches'] >= comparison_matches + 16
                and q['validation_HOST_state_relative_RMS'] <= 1.25 * baseline['validation_HOST_state_relative_RMS']
                and q['validation_HOST_direction_RMS'] <= 1.25 * baseline['validation_HOST_direction_RMS']
                and q['maximum_validation_raw_RMS_relative_change'] < .25
                and candidate.isfinite().all() and candidate.abs().max() <= 1e6 and total <= 5016268)
            trials.append(record)
            key = (q['validation_restricted_matches'], -q['validation_HOST_state_relative_RMS'])
            if diagnostic is None or key > diagnostic[0]:
                diagnostic = key, record
            if record['eligible'] and (best is None or key > best[0]):
                best = key, candidate.clone(), state.clone(), record
    print(json.dumps({'ridge': ridge, 'best_diagnostic': diagnostic[-1],
                      'selected': None if best is None else best[-1]}), flush=True)
del ev, vec, rhs, all_corrections, row_readouts, Fv

def full_readout(state, expected):
    x = (norm(state) * fg).float()
    maximum = torch.full((len(x),), -torch.inf)
    winner = torch.full_like(expected, -1); competitor = torch.full_like(maximum, -torch.inf)
    wanted = (x * E[expected]).sum(1)
    for start in range(0, len(E), 4096):
        logits = x @ E[start:start + 4096].T
        values, indices = logits.max(1); better = values > maximum
        maximum[better] = values[better]; winner[better] = start + indices[better]
        local = expected - start; chosen = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[chosen, local[chosen]] = -torch.inf
        competitor = torch.maximum(competitor, logits.amax(1))
    return {'matches': int((winner == expected).sum()), 'tokens': len(x),
            'minimum_target_margin': float((wanted - competitor).min())}

provisional = None; full = None; fmt = None; Wnew = None
if best is not None:
    provisional = best[-1]
    full = {'comparison': full_readout(comparison[nt:], truth[nt:]),
            'candidate': full_readout(best[2], truth[nt:])}
    if full['candidate']['matches'] < full['comparison']['matches'] + 16:
        best = None
if best is not None:
    Wnew = ch[dn].clone(); Wnew[rows[:, None], native[None]] = best[1]
    assert torch.equal(Wnew[:, units], ch[dn][:, units]) and torch.equal(Wnew[ordinary], ch[dn][ordinary])
    stress = torch.load(sp, weights_only=True, mmap=True); fr = expand(h, format_reference)
    assert stress['checkpoint_sha256'] == sha(format_reference)
    for name in h:
        if name not in (dn, pn):
            assert torch.equal(ch.get(name, h[name]), fr.get(name, h[name])), name
    SF = stress['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
    SB = stress['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
    expected = stress['topk_ids'][:, :, 0].reshape(-1).long()
    old_raw, new_raw = SF @ ch[dn].double().T, SF @ Wnew.double().T
    fmt = {'comparison': full_readout(SB + norm(old_raw) * old_gain, expected),
           'candidate': full_readout(SB + norm(new_raw) * gain, expected),
           'maximum_raw_RMS_relative_change': float((rms(new_raw) / rms(old_raw) - 1).abs().max())}
    assert fmt['comparison']['matches'] == len(expected)
    if fmt['candidate']['matches'] != len(expected) or fmt['candidate']['minimum_target_margin'] <= 1 or fmt['maximum_raw_RMS_relative_change'] >= .02:
        best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'original_target_reference': str(reference.relative_to(ROOT)), 'installation_base': str(install_base.relative_to(ROOT)),
          'baseline': baseline, 'training_windows': 384, 'reused_validation_windows': 128,
          'rows': rows.tolist(), 'retained_native_features': len(native), 'ridge_trials': ridges,
          'HOST_margin_fraction_trials': fractions, 'strength_trials': strengths,
          'prior': 'Original HOST Down; strong rows scaled HOSTgain/currentgain, anchor HOST raw row',
          'fixed_training_weight': '1/(HOST gap+0.1), training mean normalized, capped100',
          'targets': 'Immutable A287 joint physical training targets; no candidate residual reuse',
          'trials': trials, 'best_diagnostic': diagnostic[-1], 'provisional': provisional,
          'full_vocabulary': full, 'format': fmt, 'eligible': best is not None,
          'selected': None if best is None else best[-1],
          'scope': 'Exact recorded A123/A142 local feature/readout algebra. A259 upstream differs and requires full scoring.',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/analyze_a287_joint_projected_anchor_geometry_v1.py': sha(ROOT / 'experiments/analyze_a287_joint_projected_anchor_geometry_v1.py'),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [gp, base, reference, install_base, ap, hp, op, tp, sp,
                         format_reference, base.parent / 'circuit.pt', install_base.with_suffix('.manifest.json')]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a289v1_joint_projected_anchor_fit.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'down': Wnew, 'gain_parameter': gain_parameter, 'report': report})
print(json.dumps({'eligible': best is not None, 'selected': report['selected'],
                  'best_diagnostic': report['best_diagnostic'], 'full': full, 'format': fmt,
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
