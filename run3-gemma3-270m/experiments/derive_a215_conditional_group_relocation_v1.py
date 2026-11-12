"""Relocate layer1/2 programmes using full conditional native statistics.

Each proposed unit group gets its own donors and an independent original
HOST feature solve. Five conditional RMS/raw tradeoffs propose groups.
Ordinary post gains then have one original-native-output closed-form fit;
all programme functions and their output gains remain fixed.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state(); base = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'; ch = expand(h, base)
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified'] and cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
calibration_base = ROOT / layout['base']; ref = expand(h, calibration_base)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
del ref
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
teacher_states = {}
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(3):
    for branch, part in (('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')):
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    if layer: teacher_states[layer] = state.reshape(-1, 640).clone()
del state
nt = 192 * 64
raw_masses = [0., .01, .1, 1., 10.]
donor_counts = [64, 128, 256, 384, 512]
ridges = [1e-6, 1e-4, .01, 1., 100.]
bm = json.loads(base.with_suffix('.manifest.json').read_text())
reports, selected = {}, {}
for layer in (1, 2):
    prefix = f'model.layers.{layer}.'
    gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
    old_units = torch.tensor(layout['native_restored_programme_units'][str(layer)])
    Gprog, Uprog, Wprog = ch[gn][old_units].clone(), ch[un][old_units].clone(), ch[dn][:, old_units].clone()
    WH = h[dn].double(); gh, inherited_gain = 1 + h[pn].double(), 1 + ch[pn].double()
    fixed = Wprog.ne(0).any(1); ordinary = ~fixed
    ratio = torch.ones(640, dtype=torch.float64)
    nonzero = fixed & inherited_gain.abs().gt(1e-12)
    assert gh[fixed & ~nonzero].eq(0).all()
    ratio[nonzero] = gh[nonzero] / inherited_gain[nonzero]
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    teacher = teacher_states[layer].double()
    teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    RH = FH @ WH.T; hr = (RH.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    target = RH / hr * gh
    X = FH[:nt] / hr[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
    covariance = Z.T @ Z / nt
    regularized = (covariance + covariance.T) / 2; regularized.diagonal().add_(.001)
    precision = torch.linalg.solve(regularized, torch.eye(2048, dtype=torch.float64)); del regularized, X
    RN = RH[:nt] / hr[:nt]
    S = RN.square().sum(1, keepdim=True) + 640e-6 / hr[:nt].square()
    energy = (RN.square() * gh.square()).sum(1, keepdim=True)
    post_costs, raw_costs = [], []
    for start in range(0, 2048, 128):
        columns = torch.arange(start, min(start + 128, 2048))
        residual = (Z @ precision[:, columns]) * (sd[columns] / precision.diagonal()[columns])[None]
        W = WH[:, columns]
        dot, weighted_dot = RN @ W, RN @ (gh.square()[:, None] * W)
        wn, wg = W.square().sum(0), (W.square() * gh.square()[:, None]).sum(0)
        Sn = (S - 2 * residual * dot + residual.square() * wn).clamp_min(1e-20)
        numerator = energy - 2 * residual * weighted_dot + residual.square() * wg
        cross = energy - residual * weighted_dot
        cost = 640 * (numerator / Sn + energy / S - 2 * cross / (Sn * S).sqrt())
        if start == 0:
            literal = RN[:4, None] - residual[:4, :4, None] * W[:, :4].T[None]
            literal = (640 ** .5 * literal / Sn[:4, :4, None].sqrt() * gh
                - (640 ** .5 * RN[:4] / S[:4].sqrt() * gh)[:, None]).square().sum(-1)
            assert torch.allclose(literal, cost[:4, :4], rtol=1e-7, atol=1e-7)
        post_costs.append(cost.clamp_min(0).mean(0))
        raw_costs.append((residual.square() * wn).mean(0))
    post_cost, raw_cost = torch.cat(post_costs), torch.cat(raw_costs)
    post_cost /= post_cost.sum(); raw_cost /= raw_cost.sum()
    del Z, RN, S, energy, residual, W, dot, weighted_dot, Sn, numerator, cross, cost
    F0 = FH[nt:].clone(); F0[:, old_units] = 0
    parent_raw = F0 @ ch[dn].double().T; del F0
    parent_rms = (parent_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    def quality(raw, parameter):
        post = norm(raw) * (1 + parameter.double()); output = before[nt:] + post
        return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
            'maximum_validation_raw_RMS_relative_change': float(((raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() / parent_rms - 1).abs().max())}
    baseline = quality(parent_raw, ch[pn]); del parent_raw
    trials, best, seen = [], None, set()
    for mass in raw_masses:
        units = (post_cost + mass * raw_cost).argsort()[:len(old_units)]
        unit_key = tuple(sorted(units.tolist()))
        if unit_key in seen:
            trials.append({'raw_importance_mass': mass, 'units': units.tolist(), 'duplicate_unit_set': True}); continue
        seen.add(unit_key)
        # Full conditional prediction coefficients for this group, used
        # solely to rank its own donors. Every sparse fit targets HOST FH.
        conditional = torch.linalg.solve(precision[units][:, units], precision[units]).T
        conditional[units] = 0
        native_effect = (conditional * sd[units][None]) @ (WH[:, units] * gh[:, None]).T
        donor_order = native_effect.square().sum(1).argsort(descending=True)
        donor_order = donor_order[~torch.isin(donor_order, units)]
        G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Gprog, Uprog
        removed = FH[:, units] @ WH[:, units].T
        Fv = FH[nt:].clone(); Fv[:, units] = 0
        for count in donor_counts:
            donors = donor_order[:count]
            cov = covariance[donors][:, donors]
            ev, vec = torch.linalg.eigh((cov + cov.T) / 2); ev = ev.clamp_min(0)
            rhs = vec.T @ (covariance[donors][:, units] * sd[units][None])
            for ridge in ridges:
                beta = (vec @ (rhs / (ev + ridge)[:, None])) / sd[donors, None]
                raw = RH - removed + (FH[:, donors] @ beta) @ WH[:, units].T
                raw *= ratio[None]
                z = norm(raw)
                fitted = (z[:nt] * target[:nt]).sum(0) / z[:nt].square().sum(0).clamp_min(1e-30)
                parameter = (fitted - 1).float(); parameter[fixed] = ch[pn][fixed]
                W = WH.clone(); W[:, donors] += WH[:, units] @ beta.T
                W *= ratio[:, None]; W[:, units] = Wprog.double(); W = W.float()
                weights = {gn: G, un: U, dn: W, pn: parameter}
                changed = {name: int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) for name, value in weights.items()}
                total = bm['changed_scalars'] + sum(changed[name] - bm['per_tensor'][name] for name in changed)
                q = quality(raw[nt:], parameter)
                rec = {'raw_importance_mass': mass, 'units': units.tolist(), 'donor_count': count,
                    'donors': donors.tolist(), 'native_feature_ridge': ridge, **q,
                    'maximum_weight': max(float(v.abs().max()) for v in weights.values()),
                    'changed_scalars_per_replaced_tensor': changed, 'total_changed_scalars': total,
                    'eligible': False}
                provisional = bool(total <= 5016268 and rec['maximum_weight'] <= 1e6
                    and all(v.isfinite().all() for v in weights.values())
                    and q['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                    and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                    and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
                key = q['validation_paired_HOST_state_relative_rms']
                if provisional and (best is None or key < best[0]):
                    exact_raw = Fv @ W.double().T
                    actual = quality(exact_raw, parameter)
                    rec['rounded_weight_validation'] = actual
                    rec['rounding_raw_relative_error'] = float((exact_raw - raw[nt:]).norm() / exact_raw.norm())
                    rec['eligible'] = bool(rec['rounding_raw_relative_error'] < 1e-5
                        and actual['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                        and actual['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                        and actual['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
                    if rec['eligible']: best = actual['validation_paired_HOST_state_relative_rms'], {n: v.clone() for n, v in weights.items()}, rec
                rec['provisional_screen_pass'] = provisional
                trials.append(rec)
        print(json.dumps({'layer': layer, 'mass': mass, 'selected': None if best is None else {k: v for k, v in best[-1].items() if k != 'donors'}}), flush=True)
    reports[str(layer)] = {'baseline': baseline, 'old_units': old_units.tolist(),
        'fixed_programme_output_gain_rows': fixed.nonzero().flatten().tolist(),
        'trials': trials, 'selected': None if best is None else best[-1]}
    if best is not None: selected[str(layer)] = {'weights': best[1], 'record': best[2]}
    del FH, before, teacher, teacher_rms, RH, hr, target, covariance, precision, removed, Fv, raw, z
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'calibration_parent': str(calibration_base.relative_to(ROOT)), 'training_windows': 192, 'validation_windows': 64,
    'fixed_full_conditional_ranking_ridge': .001, 'raw_importance_mass_trials': raw_masses,
    'donor_count_trials': donor_counts, 'native_feature_ridge_trials': ridges,
    'targets': 'Original displaced HOST native features and original same-input HOST native branch; one ordinary-gain solve after each independent feature solve',
    'programme_gate_up_down_functions_and_output_gains_fixed': True,
    'source_trajectory_exact_through_layer2_for_original_base': True,
    'composition_caveat': 'Individual local same-input checks do not certify compositions or a changed earlier-layer parent; full scoring required',
    'format_caveat': 'Unknown native format features not evaluated; arithmetic functions copied bit-identically but normalization changes require full arithmetic scoring',
    'no_model_execution_gradients_iterative_residual_refit_or_answers': True,
    'layers': reports, 'selected_layers': list(selected),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, calibration_base, ap, lp, cp, hp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a215v1_conditional_group_relocation.json'; atomic_json(out, report)
if selected: atomic_torch(out.with_suffix('.pt'), {'layers': selected, 'report': report})
print(json.dumps({'selected_layers': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
