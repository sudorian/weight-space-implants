"""Preserve native layer9 Gate/Down and directly repair selected Up rows.

At unchanged recorded inputs, native Gate factors are inferred by dividing
recorded GLU products by their linear Up factors. No native activation is
evaluated. Safe divisions carry explicit roundoff bounds. Every Up solve
uses the original paired HOST state target and fixed parent normalizers.
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
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified'] and cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base)
prefix = 'model.layers.9.'
for name in h:
    if name.startswith(prefix): assert torch.equal(ch.get(name, h[name]), h[name]), name
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
indices = torch.tensor(layout['layer9_window_indices']); assert indices.tolist() == list(range(0, 256, 4))
assert torch.equal(a['inputs'], host['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
state = h['model.embed_tokens.weight'][a['inputs'][indices]] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(10):
    for branch, part in (('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')):
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw'][indices]) * (1 + h[f'model.layers.{layer}.{part}.weight'])
teacher = state.reshape(-1, 640).double(); del state
before32 = a['captures']['9.raw'].reshape(-1, 640)
P = (fnorm(before32) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'])).double()
before = before32.double()
FH = a['captures']['9.native_features'].reshape(-1, 2048).double()
un, dn, pn = [prefix + s for s in ('mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
U0, W0 = h[un].double(), h[dn].double(); gain = 1 + h[pn].double()
raw0 = FH @ W0.T; raw_rms = (raw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
current = before + raw0 / raw_rms * gain
teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
nt = 48 * 64
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
dot_allowance = 650 * eps / (1 - 650 * eps)
up = P @ U0.T
up_error = dot_allowance * (P.abs() @ U0.abs().T) + tiny
safe = (up.abs() > up_error).all(0)
safe_units = safe.nonzero().flatten()
assert len(safe_units) >= 4, ('insufficient_safe_native_factors', len(safe_units))
S = up[:, safe_units]; E0 = up_error[:, safe_units]
F = FH[:, safe_units]
phi = F / S
phi_error = F.abs() * E0 / (S.abs() * (S.abs() - E0))
phi_error += (eps / (1 - eps) * F.abs() + tiny) / (S.abs() - E0)
assert phi.isfinite().all() and phi_error.isfinite().all()
target = (teacher - current) / teacher_rms
Xbase = P / raw_rms / teacher_rms
WM = W0[:, safe_units] * gain[:, None]
unit_norm = WM.norm(dim=0).clamp_min(1e-15)
projected = target[:nt] @ (WM / unit_norm[None])
cross = Xbase[:nt].T @ (phi[:nt] * projected) / nt
input_cov = Xbase[:nt].T @ Xbase[:nt] / nt
input_cov.diagonal().add_(.001 * input_cov.diagonal().clamp_min(1e-15))
solved = torch.linalg.solve(input_cov, cross)
priority = (cross * solved).sum(0).clamp_min(0) / phi[:nt].square().mean(0).clamp_min(1e-15)
order = priority.argsort(descending=True)
declared_counts = [4, 8, 16, 32, 48]
counts = [n for n in declared_counts if n <= len(order)]
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
unit_cache = {}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
baseline = {'validation_paired_HOST_state_relative_rms': float(((current[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
    'validation_paired_HOST_direction_rms': float((norm(current[nt:]) - norm(teacher[nt:])).square().mean().sqrt())}
trials, alternatives, best = [], {}, None
print(json.dumps({'safe_native_factor_units': len(safe_units), 'counts': counts, 'baseline': baseline}), flush=True)
for count in counts:
    chosen = order[:count]; units = safe_units[chosen]
    directions = WM[:, chosen]
    gram = directions.T @ directions
    gram.diagonal().add_(.001 * gram.diagonal().mean().clamp_min(1e-15))
    coefficients = torch.linalg.solve(gram, directions.T @ target[:nt].T).T
    fits = {ridge: [] for ridge in ridges}
    for column, factor_index in enumerate(chosen.tolist()):
        Zraw = Xbase[:nt] * phi[:nt, factor_index, None]
        if factor_index not in unit_cache:
            sd = Zraw.square().mean(0).sqrt().clamp_min(1e-15)
            Z = Zraw / sd; C = Z.T @ Z / nt
            ev, vec = torch.linalg.eigh((C + C.T) / 2)
            unit_cache[factor_index] = sd, ev.clamp_min(0), vec
        sd, ev, vec = unit_cache[factor_index]
        rhs = vec.T @ ((Zraw / sd).T @ coefficients[:, column] / nt)
        for ridge in ridges:
            fits[ridge].append((vec @ (rhs / (ev + ridge))) / sd)
    local_best = None
    for ridge in ridges:
        correction = torch.stack(fits[ridge])
        for strength in strengths:
            U = h[un].clone(); U[units] = (U0[units] + strength * correction).float()
            delta = U[units].double() - U0[units]
            delta_up = P @ delta.T
            delta_feature = phi[:, chosen] * delta_up
            raw = raw0 + delta_feature @ W0[:, units].T
            state = before + norm(raw) * gain
            new_up = P @ U[units].double().T
            new_up_error = dot_allowance * (P.abs() @ U[units].double().abs().T) + tiny
            phi_upper = phi[:, chosen].abs() + phi_error[:, chosen]
            feature_error = phi_error[:, chosen] * delta_up.abs()
            feature_error += phi_upper * (new_up_error + up_error[:, units])
            feature_error += (eps / (1 - eps)) * phi_upper * (new_up.abs() + new_up_error + up[:, units].abs() + up_error[:, units]) + 2 * tiny
            raw_error = feature_error @ W0[:, units].abs().T
            uncertainty_norm = raw_error.norm(dim=1)
            raw_norm = raw.norm(dim=1)
            denominator = (raw_norm - uncertainty_norm).clamp_min(1e-30)
            state_bound = 2 * gain.abs().max() * uncertainty_norm / denominator / teacher_rms[:, 0]
            bound_rms = float(state_bound[nt:].square().mean().sqrt())
            q = {'validation_paired_HOST_state_relative_rms': float(((state[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
                'validation_paired_HOST_direction_rms': float((norm(state[nt:]) - norm(teacher[nt:])).square().mean().sqrt()),
                'maximum_validation_raw_RMS_relative_change': float(((raw[nt:].square().mean(-1, keepdim=True) + 1e-6).sqrt() / raw_rms[nt:] - 1).abs().max()),
                'validation_inferred_factor_and_Up_roundoff_state_relative_RMS_bound': bound_rms}
            changed = int((U.view(torch.int32) != h[un].view(torch.int32)).sum())
            total = bm['changed_scalars'] + changed
            rec = {'native_up_unit_count': count, 'units': units.tolist(), 'ridge': ridge, 'correction_strength': strength,
                **q, 'changed_up_scalars': changed, 'total_changed_scalars': total,
                'maximum_up_weight': float(U.abs().max()), 'maximum_up_change': float(delta.abs().max())}
            rec['eligible'] = bool(U.isfinite().all() and U.abs().max() <= 1e6 and total <= 5016268
                and q['maximum_validation_raw_RMS_relative_change'] < .05
                and q['validation_paired_HOST_state_relative_rms'] < .98 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .98 * baseline['validation_paired_HOST_direction_rms']
                and baseline['validation_paired_HOST_state_relative_rms'] - q['validation_paired_HOST_state_relative_rms'] > 2 * bound_rms)
            trials.append(rec)
            key = q['validation_paired_HOST_state_relative_rms']
            if rec['eligible'] and (local_best is None or key < local_best[0]): local_best = key, U.clone(), rec
            if rec['eligible'] and (best is None or key < best[0]): best = key, U.clone(), rec
    if local_best is not None: alternatives[str(count)] = {'up': local_best[1], 'record': local_best[2]}
    print(json.dumps({'unit_count': count, 'selected': None if local_best is None else local_best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'training_windows': 48, 'validation_windows': 16, 'window_indices': indices.tolist(),
    'native_up_unit_count_trials': declared_counts, 'executed_native_up_unit_counts': counts,
    'ridge_trials': ridges, 'correction_strength_trials': strengths,
    'fixed_input_ranking_ridge_fraction': .001, 'fixed_output_projection_ridge_fraction': .001,
    'safe_native_factor_units': len(safe_units), 'native_unit_priority_order': safe_units[order].tolist(),
    'native_factor_identity': 'Recorded native GateActivation*Up divided by linear Up at the same recorded input; no Gate activation evaluated',
    'factor_safety': 'Each chosen unit has |Up| greater than a650term float32 dot-product error bound at all4096recorded tokens; division/product intervals propagated through the Up change',
    'uncertainty_scope': 'Bound covers inferred native factors and Up dot/product rounding; it is not a full-network or all-floating-point execution certificate',
    'target': 'Original paired HOST layer9 post-MLP state at fixed original parent raw and teacher normalizers; output-direction projection and each Up row are independent original-target solves',
    'native_Gate_Down_all_norms_and_other_layers_unchanged': True,
    'format_caveat': 'No exact native-format feature capture; no complete format preservation certificate; official arithmetic score required',
    'no_model_execution_native_nonlinear_replay_gradients_iterative_fit_or_answers': True,
    'trials': trials, 'selected': None if best is None else best[-1], 'selected_counts': list(alternatives),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, ap, lp, cp, hp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a218v1_native_gate_up_repair.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'alternatives': alternatives, 'up': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
