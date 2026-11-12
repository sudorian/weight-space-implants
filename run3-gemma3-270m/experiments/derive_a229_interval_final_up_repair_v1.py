"""Native-Gate-preserving final Up repair with per-observation intervals.

Infer unchanged native Gate factors only from recorded products and linear
Up intervals. Unstable divisions are omitted from fits and bounded at
validation. Every fit targets the original paired HOST state independently.
No native activation is evaluated or differentiated.
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
prefix = 'model.layers.17.'
programme_units = torch.tensor(layout['native_restored_programme_units']['17'])
native_mask = ~torch.isin(torch.arange(2048), programme_units)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
indices = torch.arange(256)
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
top = torch.load(tp, weights_only=True, mmap=True)
assert torch.equal(a['inputs'], top['inputs'][256:])
assert torch.equal(a['inputs'], host['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
teacher = host['captures']['finalraw'].reshape(-1, 640).double()
before32 = a['captures']['17.raw'].reshape(-1, 640)
P = (fnorm(before32) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'])).double()
before = before32.double()
FH = a['captures']['17.native_features'].reshape(-1, 2048).double()
un, dn, pn = [prefix + s for s in ('mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
U0, W0 = ch[un].double(), ch[dn].double(); gain = 1 + ch[pn].double()
assert torch.equal(ch[un][native_mask], h[un][native_mask])
assert torch.equal(ch[prefix + 'mlp.gate_proj.weight'][native_mask], h[prefix + 'mlp.gate_proj.weight'][native_mask])
Factual = FH.clone(); Factual[:, programme_units] = 0
raw0 = Factual @ W0.T; del Factual; raw_rms = (raw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
current = before + raw0 / raw_rms * gain
teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
nt = 192 * 64
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
host_ids = top['topk_ids'][256:].reshape(-1, 8).long()
edited_ids = a['topk_ids'].reshape(-1, 8).long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
val_ids = torch.cat([host_ids[nt:], edited_ids[nt:]], 1)
val_truth = host_ids[nt:, 0]
val_vectors = E[val_ids].double() * fg
baseline['recorded_parent_matches'] = int((edited_ids[nt:, 0] == val_truth).sum())
baseline['reconstructed_parent_relative_error'] = float((current - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / current.norm())
assert baseline['reconstructed_parent_relative_error'] < 1e-6
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
            U = ch[un].clone(); U[units] = (U0[units] + strength * correction).float()
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
            scores = torch.einsum('nd,nkd->nk', state[nt:], val_vectors)
            winners = val_ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
            q = {'validation_restricted_matches': int((winners == val_truth).sum()),
                'validation_paired_HOST_state_relative_rms': float(((state[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
                'validation_paired_HOST_direction_rms': float((norm(state[nt:]) - norm(teacher[nt:])).square().mean().sqrt()),
                'maximum_validation_raw_RMS_relative_change': float(((raw[nt:].square().mean(-1, keepdim=True) + 1e-6).sqrt() / raw_rms[nt:] - 1).abs().max()),
                'validation_inferred_factor_and_Up_roundoff_state_relative_RMS_bound': bound_rms}
            changed = int((U.view(torch.int32) != h[un].view(torch.int32)).sum())
            total = bm['changed_scalars'] - bm['per_tensor'][un] + changed
            rec = {'native_up_unit_count': count, 'units': units.tolist(), 'ridge': ridge, 'correction_strength': strength,
                **q, 'changed_up_scalars': changed, 'total_changed_scalars': total,
                'maximum_up_weight': float(U.abs().max()), 'maximum_up_change': float(delta.abs().max()),
                'unstable_training_observations': int((~stable[:nt, chosen]).sum()),
                'unstable_validation_observations': int((~stable[nt:, chosen]).sum())}
            rec['eligible'] = bool(U.isfinite().all() and U.abs().max() <= 1e6 and total <= 5016268
                and q['maximum_validation_raw_RMS_relative_change'] < .05
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
                and q['validation_restricted_matches'] >= baseline['recorded_parent_matches'] + 8
                and baseline['validation_paired_HOST_state_relative_rms'] - q['validation_paired_HOST_state_relative_rms'] > 2 * bound_rms)
            trials.append(rec)
            key = (-q['validation_restricted_matches'], q['validation_paired_HOST_state_relative_rms'])
            if rec['eligible'] and (local_best is None or key < local_best[0]): local_best = key, U.clone(), rec
            if rec['eligible'] and (best is None or key < best[0]): best = key, U.clone(), rec, state[nt:].clone()
    if local_best is not None: alternatives[str(count)] = {'up': local_best[1], 'record': local_best[2]}
    print(json.dumps({'unit_count': count, 'selected': None if local_best is None else local_best[-1]}), flush=True)
full = None
if best is not None:
    state = (norm(best[3]) * fg).float()
    maximum = torch.full((len(state),), -torch.inf); winner = torch.full((len(state),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = state @ E[start:start + 4096].T
        values, indices = logits.max(1); improve = values > maximum
        maximum[improve] = values[improve]; winner[improve] = start + indices[improve]
    full = {'tokens': len(state), 'parent_recorded_matches': baseline['recorded_parent_matches'],
        'candidate_matches': int((winner == val_truth).sum()), 'required_extra_matches': 8}
    if full['candidate_matches'] < full['parent_recorded_matches'] + 8:
        best = None; alternatives = {}
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'training_windows': 192, 'validation_windows': 64, 'window_indices': indices.tolist(),
    'native_up_unit_count_trials': declared_counts, 'executed_native_up_unit_counts': counts,
    'ridge_trials': ridges, 'correction_strength_trials': strengths,
    'fixed_input_ranking_ridge_fraction': .001, 'fixed_output_projection_ridge_fraction': .001,
    'native_factor_units_with_at_least_99percent_stable_training_observations': len(safe_units),
    'minimum_training_stable_fraction': .99, 'stable_division_rule': '|Up|>2times Up dot error',
    'factor_interval_intersection_inconsistencies': interval_inconsistencies, 'native_unit_priority_order': safe_units[order].tolist(),
    'native_factor_identity': 'Recorded native GateActivation*Up divided by linear Up at the same recorded input; no Gate activation evaluated',
    'factor_safety': 'Per-observation division intervals intersected with input-sign and absolute Gate bounds. Near-zero Up observations omitted from fitting, retained with interval uncertainty in validation.650-term float32 dot bounds and coordinatewise output gain propagation.',
    'uncertainty_scope': 'Bound covers inferred native factors and Up dot/product rounding; it is not a full-network or all-floating-point execution certificate',
    'target': 'Original paired HOST final state at fixed original parent raw and teacher normalizers; output-direction projection and each Up row are independent original-target solves',
    'native_Gate_Down_all_norms_and_other_layers_unchanged': True,
    'format_caveat': 'No exact native-format feature capture; no complete format preservation certificate; official arithmetic score required',
    'no_model_execution_native_nonlinear_replay_gradients_iterative_fit_or_answers': True,
    'trials': trials, 'full_vocabulary_validation': full, 'selected': None if best is None else best[2], 'selected_counts': list(alternatives),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', ap, lp, cp, hp, tp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a229v1_interval_final_up_repair.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'alternatives': alternatives, 'up': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
