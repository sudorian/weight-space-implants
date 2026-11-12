"""Native-Gate-preserving branch Up repair with per-observation intervals.

Infer unchanged native Gate factors only from recorded products and linear
Up intervals. Unstable divisions are omitted from fits and bounded at
validation. Every fit targets the original paired HOST state independently.
No native activation is evaluated or differentiated.
"""
import argparse, json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
parser = argparse.ArgumentParser(); parser.add_argument('--layer', type=int, required=True, choices=list(range(10)))
args = parser.parse_args(); layer = args.layer
h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified'] and cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base)
prefix = f'model.layers.{layer}.'
programme_units = torch.tensor(layout['native_restored_programme_units'][str(layer)], dtype=torch.long)
native_mask = ~torch.isin(torch.arange(2048), programme_units)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
indices = torch.tensor(layout['layer9_window_indices'], dtype=torch.long) if layer == 9 else torch.arange(256)
training_windows = 48 if layer == 9 else 192
validation_windows = 16 if layer == 9 else 64
assert torch.equal(a['inputs'], host['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
host_state_before = h['model.embed_tokens.weight'][a['inputs'][indices]] * torch.tensor(640 ** .5, dtype=torch.float32)
for previous_layer in range(layer + 1):
    for branch, part in (('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')):
        host_state_before = host_state_before + fnorm(host['captures'][f'{previous_layer}.{branch}.raw'][indices]) * (1 + h[f'model.layers.{previous_layer}.{part}.weight'])
teacher = host_state_before.reshape(-1, 640).double(); del host_state_before
before32 = a['captures'][f'{layer}.raw'].reshape(-1, 640)
P = (fnorm(before32) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'])).double()
before = before32.double()
FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
un, dn, pn = [prefix + s for s in ('mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
U0, W0 = ch.get(un, h[un]).double(), ch.get(dn, h[dn]).double(); gain = 1 + ch.get(pn, h[pn]).double()
assert torch.equal(ch.get(un, h[un])[native_mask], h[un][native_mask])
assert torch.equal(ch.get(prefix + 'mlp.gate_proj.weight', h[prefix + 'mlp.gate_proj.weight'])[native_mask], h[prefix + 'mlp.gate_proj.weight'][native_mask])
Factual = FH.clone(); Factual[:, programme_units] = 0
raw0 = Factual @ W0.T; del Factual; raw_rms = (raw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
current = before + raw0 / raw_rms * gain
teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
nt = training_windows * 64
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
dot_allowance = 650 * eps / (1 - 650 * eps)
up = P @ U0.T
up_error = dot_allowance * (P.abs() @ U0.abs().T) + tiny
stable_all = up.abs() > 2 * up_error
training_stable_fraction = stable_all[:nt].double().mean(0)
safe_units = (native_mask & (training_stable_fraction >= .99)).nonzero().flatten()
assert len(safe_units) >= 48
phi = torch.empty(len(P), len(safe_units), dtype=torch.float64)
phi_error = torch.empty_like(phi)
stable = stable_all[:, safe_units]; del stable_all
interval_inconsistencies = 0
for start in range(0, len(safe_units), 128):
    subset = safe_units[start:start + 128]
    S = up[:, subset]; E0 = up_error[:, subset]; FF = FH[:, subset]
    gate_rows = h[prefix + 'mlp.gate_proj.weight'][subset].double()
    gg = P @ gate_rows.T
    ge = dot_allowance * (P.abs() @ gate_rows.abs().T) + tiny
    # GELU has the input's sign and magnitude no greater than its input.
    # A small explicit allowance covers the scalar activation rounding.
    lower = torch.minimum(torch.zeros_like(gg), gg - ge) * (1 + 4 * eps) - tiny
    upper = torch.maximum(torch.zeros_like(gg), gg + ge) * (1 + 4 * eps) + tiny
    safe_division = S.abs() > E0
    ferr = eps / (1 - eps) * FF.abs() + tiny
    denom0 = torch.where(safe_division, S - E0, torch.ones_like(S))
    denom1 = torch.where(safe_division, S + E0, torch.ones_like(S))
    r0, r1, r2, r3 = (FF - ferr) / denom0, (FF - ferr) / denom1, (FF + ferr) / denom0, (FF + ferr) / denom1
    ratio_lower = torch.minimum(torch.minimum(r0, r1), torch.minimum(r2, r3))
    ratio_upper = torch.maximum(torch.maximum(r0, r1), torch.maximum(r2, r3))
    intersect_lower, intersect_upper = torch.maximum(lower, ratio_lower), torch.minimum(upper, ratio_upper)
    inconsistent = safe_division & (intersect_lower > intersect_upper)
    interval_inconsistencies += int(inconsistent.sum())
    use_ratio = safe_division & ~inconsistent
    lower = torch.where(use_ratio, intersect_lower, lower)
    upper = torch.where(use_ratio, intersect_upper, upper)
    phi[:, start:start + len(subset)] = (lower + upper) / 2
    phi_error[:, start:start + len(subset)] = (upper - lower) / 2
assert phi.isfinite().all() and phi_error.isfinite().all()
del FH, S, E0, FF, gg, ge, lower, upper, denom0, denom1, r0, r1, r2, r3, ratio_lower, ratio_upper, intersect_lower, intersect_upper
target = (teacher - current) / teacher_rms
Xbase = P / raw_rms / teacher_rms
WM = W0[:, safe_units] * gain[:, None]
unit_norm = WM.norm(dim=0).clamp_min(1e-15)
projected = target[:nt] @ (WM / unit_norm[None])
cross = Xbase[:nt].T @ (phi[:nt] * stable[:nt] * projected) / nt
input_cov = Xbase[:nt].T @ Xbase[:nt] / nt
input_cov.diagonal().add_(.001 * input_cov.diagonal().clamp_min(1e-15))
solved = torch.linalg.solve(input_cov, cross)
priority = (cross * solved).sum(0).clamp_min(0) / (phi[:nt].square() * stable[:nt]).mean(0).clamp_min(1e-15)
order = priority.argsort(descending=True)
declared_counts = [4, 8, 16, 32, 48]
counts = [n for n in declared_counts if n <= len(order)]
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
unit_cache = {}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
baseline = {'validation_paired_HOST_state_relative_rms': float(((current[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
    'validation_paired_HOST_direction_rms': float((norm(current[nt:]) - norm(teacher[nt:])).square().mean().sqrt())}
baseline['all_window_state_error'] = float(((current - teacher) / teacher_rms).square().mean().sqrt())
assert abs(baseline['all_window_state_error'] - cert['layers'][str(layer)]['paired_original_HOST_state_relative_rms']) < 1e-7
trials, alternatives, best = [], {}, None
print(json.dumps({'native_factor_units_with_at_least_99percent_stable_training_observations': len(safe_units),
    'minimum_training_stable_fraction': .99, 'stable_division_rule': '|Up|>2times Up dot error',
    'factor_interval_intersection_inconsistencies': interval_inconsistencies, 'counts': counts, 'baseline': baseline}), flush=True)
for count in counts:
    chosen = order[:count]; units = safe_units[chosen]
    directions = WM[:, chosen]
    gram = directions.T @ directions
    gram.diagonal().add_(.001 * gram.diagonal().mean().clamp_min(1e-15))
    coefficients = torch.linalg.solve(gram, directions.T @ target[:nt].T).T
    fits = {ridge: [] for ridge in ridges}
    for column, factor_index in enumerate(chosen.tolist()):
        Zraw = Xbase[:nt] * (phi[:nt, factor_index] * stable[:nt, factor_index])[:, None]
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
            U = ch.get(un, h[un]).clone(); U[units] = (U0[units] + strength * correction).float()
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
            # Coordinatewise gain propagation avoids the vacuous global
            #1e6-gain bound when repaired native columns are compensated.
            post_bound = 640 ** .5 * gain.abs() * (raw_error / denominator[:, None]
                + raw.abs() * (uncertainty_norm / (raw_norm.clamp_min(1e-30) * denominator))[:, None])
            state_bound = post_bound.square().mean(1).sqrt() / teacher_rms[:, 0]
            bound_rms = float(state_bound[nt:].square().mean().sqrt())
            q = {                'validation_paired_HOST_state_relative_rms': float(((state[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
                'validation_paired_HOST_direction_rms': float((norm(state[nt:]) - norm(teacher[nt:])).square().mean().sqrt()),
                'maximum_validation_raw_RMS_relative_change': float(((raw[nt:].square().mean(-1, keepdim=True) + 1e-6).sqrt() / raw_rms[nt:] - 1).abs().max()),
                'validation_inferred_factor_and_Up_roundoff_state_relative_RMS_bound': bound_rms}
            changed = int((U.view(torch.int32) != h[un].view(torch.int32)).sum())
            total = bm['changed_scalars'] - bm['per_tensor'].get(un, 0) + changed
            rec = {'native_up_unit_count': count, 'units': units.tolist(), 'ridge': ridge, 'correction_strength': strength,
                **q, 'changed_up_scalars': changed, 'total_changed_scalars': total,
                'maximum_up_weight': float(U.abs().max()), 'maximum_up_change': float(delta.abs().max()),
                'unstable_training_observations': int((~stable[:nt, chosen]).sum()),
                'unstable_validation_observations': int((~stable[nt:, chosen]).sum())}
            rec['eligible'] = bool(U.isfinite().all() and U.abs().max() <= 1e6 and total <= 5016268
                and q['maximum_validation_raw_RMS_relative_change'] < .05
                and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms']
                and baseline['validation_paired_HOST_state_relative_rms'] - q['validation_paired_HOST_state_relative_rms'] > 2 * bound_rms)
            trials.append(rec)
            key = q['validation_paired_HOST_state_relative_rms']
            if rec['eligible'] and (local_best is None or key < local_best[0]): local_best = key, U.clone(), rec
            if rec['eligible'] and (best is None or key < best[0]): best = key, U.clone(), rec, state[nt:].clone()
    if local_best is not None: alternatives[str(count)] = {'up': local_best[1], 'record': local_best[2]}
    print(json.dumps({'unit_count': count, 'selected': None if local_best is None else local_best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'layer': layer, 'training_windows': training_windows, 'validation_windows': validation_windows, 'window_indices': indices.tolist(),
    'native_up_unit_count_trials': declared_counts, 'executed_native_up_unit_counts': counts,
    'ridge_trials': ridges, 'correction_strength_trials': strengths,
    'fixed_input_ranking_ridge_fraction': .001, 'fixed_output_projection_ridge_fraction': .001,
    'native_factor_units_with_at_least_99percent_stable_training_observations': len(safe_units),
    'minimum_training_stable_fraction': .99, 'stable_division_rule': '|Up|>2times Up dot error',
    'factor_interval_intersection_inconsistencies': interval_inconsistencies, 'native_unit_priority_order': safe_units[order].tolist(),
    'native_factor_identity': 'Recorded native GateActivation*Up divided by linear Up at the same recorded input; no Gate activation evaluated',
    'factor_safety': 'Per-observation division intervals intersected with input-sign and absolute Gate bounds. Near-zero Up observations omitted from fitting, retained with interval uncertainty in validation.650-term float32 dot bounds and coordinatewise output gain propagation.',
    'uncertainty_scope': 'Bound covers inferred native factors and Up dot/product rounding; it is not a full-network or all-floating-point execution certificate',
    'target': 'Original paired HOST post-MLP state at the selected layer at fixed original parent raw and teacher normalizers; output-direction projection and each Up row are independent original-target solves',
    'native_Gate_Down_all_norms_and_other_layers_unchanged': True,
    'format_caveat': 'No exact native-format feature capture; no complete format preservation certificate; official arithmetic score required',
    'no_model_execution_native_nonlinear_replay_gradients_iterative_fit_or_answers': True,
    'local_screen': 'At least5percent paired-state and direction gain, state improvement>twice uncertainty bound, raw RMS change<5percent, edit/weight limits',
    'trials': trials, 'selected': None if best is None else best[2], 'selected_counts': list(alternatives),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', ap, lp, cp, hp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / f'artifacts/a230v1_interval_branch_up_repair_l{layer}.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'alternatives': alternatives, 'up': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
