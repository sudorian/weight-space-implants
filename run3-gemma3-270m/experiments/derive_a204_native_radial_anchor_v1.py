"""One-shot native raw-anchor repair from the original fixed-parent numerator.

For each quiet output coordinate, keep every other numerator fixed and
project the original HOST target onto its positive radial direction. The
minimum possible RMS gives an analytic feasibility cap; the anchor then
supplies the remaining raw energy. One closed-form feature solve approximates
that target. No candidate updates the target/reference, no native activation
is replayed, and no gradient or addition answer enters the construction.
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
dp = ROOT / 'artifacts/a201v1_native_rms_error.json'
layout, certificate, diagnostic = [json.loads(p.read_text()) for p in (lp, cp, dp)]
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
bm = json.loads(base.with_suffix('.manifest.json').read_text())
nt = 192 * 64
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

teacher_states = {}
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
layers = [layer for layer in layout['layers'] if layer != 9]
for layer in range(18):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    if layer in layers:
        teacher_states[layer] = state.reshape(-1, 640).clone()
assert torch.equal(state, host['captures']['finalraw'])
del state
reports, selected = {}, {}
for layer in layers:
    prefix = f'model.layers.{layer}.'
    dn, pn = prefix + 'mlp.down_proj.weight', prefix + 'post_feedforward_layernorm.weight'
    units = torch.tensor(layout['native_restored_programme_units'][str(layer)])
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    F = FH.clone(); F[:, units] = 0
    W0, WH = ch[dn].double(), h[dn].double()
    gain, gh = 1 + ch[pn].double(), 1 + h[pn].double()
    RH, R0 = FH @ WH.T, F @ W0.T
    target = norm(RH) * gh
    reference = (R0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    numerator = R0 * gain
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    teacher = teacher_states[layer].double()
    trms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    # Large existing raw energy and small output gain make a useful carrier.
    priority = (gain.abs() + 1e-6).square() / R0[:nt].square().mean(0).clamp_min(1e-15)
    priority[gain.abs() > 1] = torch.inf
    order = priority.argsort()
    anchors = order[torch.isfinite(priority[order])][:5]
    baseline_post = norm(R0[nt:]) * gain
    baseline_output = before[nt:] + baseline_post
    baseline = {
        'validation_native_branch_rms': float((baseline_post - target[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((baseline_output - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(baseline_output) - norm(teacher[nt:])).square().mean().sqrt())}
    assert abs(baseline['validation_native_branch_rms'] - diagnostic['layers'][str(layer)]['validation_current_native_branch_rms']) < 1e-9
    if not len(anchors):
        reports[str(layer)] = {'baseline': baseline, 'selected': None, 'reason': 'No output gain at most1 in magnitude'}
        del FH, F, RH, R0, target, reference, numerator, before, teacher, trms
        continue
    source_targets, target_reports = [], []
    total_energy = R0.square().sum(1)
    total_cross = (numerator * target).sum(1)
    total_numerator_energy = numerator.square().sum(1)
    for anchor in anchors.tolist():
        other_energy = (total_energy - R0[:, anchor].square()).clamp_min(0)
        cross = total_cross - numerator[:, anchor] * target[:, anchor]
        squared = total_numerator_energy - numerator[:, anchor].square()
        assert cross.min() > 0 and squared.min() > 0
        desired_alpha = cross / squared
        maximum_alpha = (640 / (other_energy + 640e-6)).sqrt()
        feasible_alpha = torch.minimum(desired_alpha, maximum_alpha)
        # Ratio>=1 makes the nonnegative remainder explicit, without
        # discarding or clipping observations with infeasible desired RMS.
        room = ((maximum_alpha / feasible_alpha).square() - 1) * (other_energy + 640e-6)
        assert room.min() >= 0 and room.isfinite().all()
        sign = torch.where(R0[:, anchor] < 0, -torch.ones_like(room), torch.ones_like(room))
        anchor_target = sign * room.sqrt()
        source_targets.append((anchor_target - R0[:, anchor]) / reference[:, 0])
        target_reports.append({'anchor': anchor, 'output_gain': float(gain[anchor]),
            'training_desired_radial_factor_capped': int((desired_alpha[:nt] > maximum_alpha[:nt]).sum()),
            'validation_desired_radial_factor_capped': int((desired_alpha[nt:] > maximum_alpha[nt:]).sum()),
            'target_anchor_raw_rms': float(anchor_target.square().mean().sqrt()),
            'target_relative_anchor_change_rms': float(source_targets[-1].square().mean().sqrt()),
            'sign_policy': 'Keep the original parent raw-anchor sign at each recorded token',
            'radial_optimum_scope': 'Exact for zero output gain; ignores the quiet coordinate output term when its gain is nonzero, and actual branch error is checked below'})
    Y = torch.stack(source_targets, 1)
    X = F[:, native] / reference
    sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-15)
    Z = X[:nt] / sd
    C = Z.T @ Z / nt
    ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    rhs = V.T @ (Z.T @ Y[:nt] / nt)
    del Z, C, source_targets, Y
    trials, best = [], None
    Fv = F[nt:, native]
    for ridge in ridges:
        correction = (V @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
        for column, anchor in enumerate(anchors.tolist()):
            for strength in strengths:
                row_weights = ch[dn][anchor].clone()
                row_weights[native] = (W0[anchor, native] + strength * correction[:, column]).float()
                assert torch.equal(row_weights[units], ch[dn][anchor, units])
                R = R0[nt:].clone(); R[:, anchor] = Fv @ row_weights[native].double()
                post = norm(R) * gain; output = before[nt:] + post
                count = bm['changed_scalars']
                count -= int((ch[dn][anchor].view(torch.int32) != h[dn][anchor].view(torch.int32)).sum())
                count += int((row_weights.view(torch.int32) != h[dn][anchor].view(torch.int32)).sum())
                rec = {'anchor': anchor, 'ridge': ridge, 'correction_strength': strength,
                    'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
                    'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
                    'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
                    'maximum_validation_relative_raw_RMS_change': float(((R.square().mean(-1, keepdim=True) + 1e-6).sqrt() / reference[nt:] - 1).abs().max()),
                    'maximum_anchor_weight': float(row_weights.abs().max()), 'changed_scalars_if_applied_alone': count}
                rec['eligible'] = bool(row_weights.isfinite().all() and row_weights.abs().max() <= 1e6 and count <= 5016268
                    and rec['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                    and rec['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                    and rec['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms'])
                trials.append(rec)
                if rec['eligible'] and (best is None or rec['validation_native_branch_rms'] < best[0]):
                    best = rec['validation_native_branch_rms'], row_weights.clone(), rec
        print(json.dumps({'layer': layer, 'ridge': ridge, 'best': None if best is None else best[-1]}), flush=True)
    reports[str(layer)] = {'baseline': baseline, 'anchor_order': anchors.tolist(), 'anchor_target_diagnostics': target_reports,
        'trials': trials, 'selected': None if best is None else best[-1]}
    if best is not None:
        value = ch[dn].clone(); value[best[-1]['anchor']] = best[1]
        assert torch.equal(value[:, units], ch[dn][:, units])
        selected[dn] = value
    del FH, F, RH, R0, target, reference, numerator, before, teacher, trms, X, ev, V, rhs, Fv
    del correction, row_weights, R, post, output

total = bm['changed_scalars']
for name, value in selected.items():
    total += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'layers': reports,
    'ridge_trials': ridges, 'correction_strength_trials': strengths, 'maximum_anchor_choices_per_layer': 5,
    'fixed_quiet_output_gain_limit': 1., 'training_windows': 192, 'validation_windows': 64,
    'selected_tensor_names': list(selected), 'combined_changed_scalars': total,
    'target': 'One analytic positive radial projection of the original HOST native branch onto the original fixed parent numerator outside the chosen anchor',
    'reference_never_updated_from_candidate': True, 'only_native_coefficients_of_one_Down_row_per_layer_change': True,
    'programme_G_Up_Down_columns_and_all_gains_fixed': True,
    'format_caveat': 'No exact current-parent native format features after its O3 repair; full official arithmetic scoring remains required',
    'composition_caveat': 'Each accepted branch independently repairs original HOST native behavior at recorded parent inputs; composed downstream trajectories still require official scoring',
    'no_answers_gradients_native_activation_replay_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, ap, lp, cp, hp, dp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a204v1_native_radial_anchor.json'; atomic_json(out, report)
if selected and total <= 5016268:
    atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected_tensor_names': list(selected), 'combined_changed_scalars': total,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
