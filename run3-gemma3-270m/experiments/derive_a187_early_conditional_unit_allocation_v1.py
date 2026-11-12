"""Relocate early arithmetic functions using exact native-removal RMS costs.

Certified original native features at the current parent inputs provide all
targets. Conditional feature residuals are projected once from fixed donor
units. A norm/dot-product identity measures each removed feature's full RMS
branch error, without native activation replay or differentiation. Five
raw-energy tradeoffs propose allocations. Every reconstruction starts from
original HOST targets and the same fixed parent normalizer.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / meta['base']; assert sha(base) == meta['base_sha256']
ch = expand(h, base)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
bm = json.loads(base.with_suffix('.manifest.json').read_text())
nt = 192 * 64
masses = [0., .01, .1, 1., 10.]
ridges = [1e-6, 1e-5, 1e-4, .001, .01]
expected_donors = [64, 512, 512, 512, 128, 128, 512, 512]
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

teacher_states = {}
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(8):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    teacher_states[layer] = state.reshape(-1, 640).clone()
del state
reports, selected = {}, {}
for layer in range(8):
    prefix = f'model.layers.{layer}.'
    gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
    old_units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    mask = (ch[dn].view(torch.int32) != h[dn].view(torch.int32)).clone(); mask[:, old_units] = False
    pool = (mask.sum(1) > 1000).nonzero().flatten()
    ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
    donors = (mask[ordinary].sum(0) > int(ordinary.sum()) / 2).nonzero().flatten()
    assert len(donors) == expected_donors[layer] and not torch.isin(old_units, donors).any()
    available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
    Gprog, Uprog, Wprog = ch[gn][old_units].clone(), ch[un][old_units].clone(), ch[dn][:, old_units].clone()
    WH = h[dn].double(); gh = 1 + h[pn].double(); gain = 1 + ch[pn].double()
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    teacher = teacher_states[layer].double()
    teacher_rms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    RH = FH @ WH.T; hr = (RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
    target = RH / hr * gh
    F0 = FH.clone(); F0[:, old_units] = 0
    R0 = F0 @ ch[dn].double().T
    reference = (R0.square().mean(1, keepdim=True) + 1e-6).sqrt()
    del F0, R0

    # Conditional native-feature fit; use fixed original-HOST normalizers.
    D = FH[:nt, donors] / hr[:nt]
    P = FH[:nt, available] / hr[:nt]
    sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
    C = Z.T @ Z / nt; C.diagonal().add_(.001)
    fit = torch.linalg.solve(C, Z.T @ P / nt) / sd[:, None]
    residual = P - D @ fit
    RN = RH[:nt] / hr[:nt]
    S = RN.square().sum(1, keepdim=True) + 640e-6 / hr[:nt].square()
    weighted_energy = (RN.square() * gh.square()).sum(1, keepdim=True)
    post_cost, raw_cost = [], []
    for start in range(0, len(available), 128):
        columns = available[start:start + 128]
        W = WH[:, columns]; e = residual[:, start:start + 128]
        dot = RN @ W; weighted_dot = RN @ (gh.square()[:, None] * W)
        wn = W.square().sum(0); wg = (W.square() * gh.square()[:, None]).sum(0)
        Sn = (S - 2 * e * dot + e.square() * wn).clamp_min(1e-20)
        numerator = weighted_energy - 2 * e * weighted_dot + e.square() * wg
        cross = weighted_energy - e * weighted_dot
        cost = 640 * (numerator / Sn + weighted_energy / S - 2 * cross / (Sn * S).sqrt())
        if start == 0:
            # Check the closed-form identity against literal vector RMS
            # arithmetic on four recorded states and four native columns.
            altered = RN[:4, None] - e[:4, :4, None] * W[:, :4].T[None]
            native_post = 640 ** .5 * RN[:4] / S[:4].sqrt() * gh
            altered_post = 640 ** .5 * altered / Sn[:4, :4, None].sqrt() * gh
            explicit = (altered_post - native_post[:, None]).square().sum(-1)
            assert torch.allclose(explicit, cost[:4, :4], atol=1e-7, rtol=1e-7)
            del altered, native_post, altered_post, explicit
        post_cost.append(cost.clamp_min(0).mean(0))
        raw_cost.append((e.square() * wn).mean(0))
    post_cost = torch.cat(post_cost); raw_cost = torch.cat(raw_cost)
    post_cost /= post_cost.sum(); raw_cost /= raw_cost.sum()
    del D, P, Z, C, fit, residual, RN, S, weighted_energy, W, e, dot, weighted_dot, Sn, numerator, cross, cost

    # Fixed-parent-normalizer original-target native reconstruction.
    design = FH / reference
    standard = design[:nt].square().mean(0).sqrt().clamp_min(1e-12)
    Z = design[:nt] / standard; cov = Z.T @ Z / nt; del Z
    Cd = cov[donors][:, donors]; ed, Vd = torch.linalg.eigh((Cd + Cd.T) / 2); ed = ed.clamp_min(0)
    Zd = design[:nt, donors] / standard[donors]
    zero_gain = gain == 0
    raw_target = target[:nt] / torch.where(zero_gain, torch.ones_like(gain), gain)
    raw_target[:, zero_gain] = RH[:nt, zero_gain] / reference[:nt]

    def quality(W, units):
        F = FH[nt:].clone(); F[:, units] = 0
        R = F @ W.double().T; post = norm(R) * gain; output = before[nt:] + post
        return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
            'maximum_relative_raw_rms_change': float(((R.square().mean(1, keepdim=True) + 1e-6).sqrt() / reference[nt:] - 1).abs().max())}

    baseline = quality(ch[dn], old_units)
    seen, trials, best = set(), [], None
    for mass in masses:
        units = available[(post_cost + mass * raw_cost).argsort()[:len(old_units)]]
        key = tuple(sorted(units.tolist()))
        if key in seen:
            trials.append({'raw_energy_mass': mass, 'units': units.tolist(), 'duplicate_unit_set': True}); continue
        seen.add(key)
        retained = int(torch.isin(units, old_units).sum())
        if retained == len(old_units):
            trials.append({'raw_energy_mass': mass, 'units': units.tolist(), 'unchanged_unit_set': True}); continue
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
        Cn = cov[native][:, native]; en, Vn = torch.linalg.eigh((Cn + Cn.T) / 2); en = en.clamp_min(0)
        removed = design[:nt, units] @ WH[:, units].T
        original_delta = raw_target - RH[:nt] / reference[:nt] + removed
        rhsd = Vd.T @ (Zd.T @ original_delta / nt)
        Zn = design[:nt, native] / standard[native]
        rhsn = Vn.T @ (Zn.T @ raw_target[:, pool] / nt)
        G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Gprog, Uprog
        for ridge in ridges:
            W = WH.clone(); W[:, units] = 0
            delta = (Vd @ (rhsd / (ed[:, None] + ridge))) / standard[donors, None]
            W[:, donors] += delta.T
            fit = (Vn @ (rhsn / (en[:, None] + ridge))) / standard[native, None]
            W[pool[:, None], native[None]] = fit.T
            W[:, units] = Wprog.double(); W = W.float()
            q = quality(W, units)
            count = bm['changed_scalars']
            for name, value in [(gn, G), (un, U), (dn, W)]:
                count += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
            row = {'raw_energy_mass': mass, 'native_ridge': ridge, 'units': units.tolist(), 'old_units_retained': retained,
                **q, 'changed_scalars_if_applied_alone': count, 'maximum_down_weight': float(W.abs().max())}
            row['eligible'] = bool(count <= 5016268 and W.isfinite().all() and W.abs().max() <= 1e6
                and q['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_native_branch_rms'] < best[0]):
                best = q['validation_native_branch_rms'], G.clone(), U.clone(), W.clone(), row
        del Cn, en, Vn, removed, original_delta, rhsd, Zn, rhsn, delta, fit
    record = {'baseline': baseline, 'old_units': old_units.tolist(), 'donors': donors.tolist(), 'strong_rows': pool.tolist(),
        'conditional_feature_ridge': .001, 'raw_energy_mass_trials': masses, 'native_ridge_trials': ridges,
        'cost_identity': 'Exact squared RMS branch error after subtracting one conditional native feature; no derivative',
        'trials': trials, 'selected': None if best is None else best[-1], 'accepted_local_candidate': best is not None}
    reports[str(layer)] = record
    if best is not None: selected[gn], selected[un], selected[dn] = best[1:4]
    print(json.dumps({'layer': layer, 'baseline': baseline, 'selected': record['selected']}), flush=True)
    del FH, before, teacher, teacher_rms, RH, hr, target, reference, design, cov, Cd, ed, Vd, Zd, raw_target

count = bm['changed_scalars']
for name, value in selected.items(): count += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'training_windows': 192, 'validation_windows': 64,
    'layers': reports, 'selected_tensor_names': list(selected), 'combined_changed_scalars': count,
    'target': 'Original same-input HOST native branch; parent RMS held fixed for every solve',
    'arithmetic_functions_copied_bit_exactly_and_gains_fixed': True,
    'format_caveat': 'Restored native format features are unobserved; exact current-parent format states unavailable. Official score required.',
    'no_answers_gradients_native_activation_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ap, lp, cp, hp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a187v1_early_conditional_unit_allocation.json'
atomic_json(out, report)
if selected and count <= 5016268: atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected': list(selected), 'combined_changed_scalars': count, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
