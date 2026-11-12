"""Restore early native RMS with quiet-coordinate energy anchors.

This applies the A141 energy identity to the current early layers0..7,
using certified exact-input HOST features. Each layer starts from HOST,
reconstructs its displaced native functions once, fits strong-row numerators
at the fixed HOST RMS, and projects a square-root energy remainder into one
quiet output coordinate. All programme columns and their gains stay fixed.
Original HOST targets are never replaced by a candidate residual target.
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
    dn = prefix + 'mlp.down_proj.weight'; pn = prefix + 'post_feedforward_layernorm.weight'
    units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
    mask = (ch[dn].view(torch.int32) != h[dn].view(torch.int32)).clone()
    mask[:, units] = False
    pool = (mask.sum(1) > 1000).nonzero().flatten()
    ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
    donors = (mask[ordinary].sum(0) > int(ordinary.sum()) / 2).nonzero().flatten()
    assert len(donors) == expected_donors[layer]
    assert not torch.isin(units, donors).any()
    programme = ch[dn][:, units].clone()
    fixed_gain_rows = programme.ne(0).any(1); fixed_gain_rows[pool] = True
    parameter = h[pn].clone(); parameter[fixed_gain_rows] = ch[pn][fixed_gain_rows]
    gain = 1 + parameter.double()
    strong = pool[gain[pool] != 0]
    WH = h[dn].double(); gh = 1 + h[pn].double(); gb = 1 + ch[pn].double()
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    F = FH.clone(); F[:, units] = 0
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    teacher = teacher_states[layer].double()
    trms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    RH = FH @ WH.T
    hr = (RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
    target = RH / hr * gh

    # One donor projection of the original native features displaced by the
    # unchanged hand programme. It is independent of every anchor trial.
    X = FH[:nt, donors] / hr[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / sd
    C = Z.T @ Z / nt; C.diagonal().add_(.001)
    repair = torch.linalg.solve(C, Z.T @ (FH[:nt, units] / hr[:nt]) / nt) / sd[:, None]
    W0 = WH.clone(); W0[:, donors] += WH[:, units] @ repair.T
    zero_gain = gain == 0
    # A programme normalization anchor can have zero output gain even when
    # HOST has a small nonzero gain there. Preserve the zero; its unavoidable
    # missing native output remains part of the measured branch error.
    ratio = torch.ones_like(gain); ratio[~zero_gain] = gh[~zero_gain] / gain[~zero_gain]
    W0 *= ratio[:, None]
    W0[:, units] = programme.double()

    XN = F[:, native] / hr
    sdn = XN[:nt].square().mean(0).sqrt().clamp_min(1e-12)
    ZN = XN[:nt] / sdn
    C = ZN.T @ ZN / nt
    ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    rhs = V.T @ (ZN.T @ (target[:nt, strong] / gain[strong]) / nt)
    strong_trials, strong_best = [], None
    for ridge in ridges:
        fit = (V @ (rhs / (ev[:, None] + ridge))) / sdn[:, None]
        error = (XN[nt:] @ fit) * gain[strong] - target[nt:, strong]
        error_rms = float(error.square().mean().sqrt()) if len(strong) else 0.
        strong_trials.append({'ridge': ridge, 'validation_fixed_HOST_normalizer_strong_rms': error_rms})
        if strong_best is None or error_rms < strong_best[0]: strong_best = error_rms, fit.clone(), ridge
    W0[strong[:, None], native[None]] = strong_best[1].T
    W0 = W0.float().double()
    R0 = F @ W0.T
    # Fixed original-HOST energy statistics rank five quiet coordinates.
    importance = (gh.abs() + 1e-4).square() / RH[:nt].square().mean(0).clamp_min(1e-12)
    importance[strong] = torch.inf
    # A fixed arithmetic gain must remain tiny for a usable quiet anchor.
    importance[gain.abs() > 1] = torch.inf
    order = importance.argsort()
    anchors = order[torch.isfinite(importance[order])][:5]

    def quality(W, g):
        R = F[nt:] @ W.double().T
        post = norm(R) * g
        output = before[nt:] + post
        return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
            'validation_relative_raw_rms_error': float(((R.square().mean(1, keepdim=True) + 1e-6).sqrt() / hr[nt:] - 1).square().mean().sqrt()),
            'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt())}

    baseline = quality(ch[dn], gb)
    trials, best = [], None
    for anchor_tensor in anchors:
        anchor = int(anchor_tensor)
        other_energy = R0.square().sum(1) - R0[:, anchor].square()
        room = RH.square().sum(1) - other_energy
        infeasible_train = int((room[:nt] < 0).sum()); infeasible_validation = int((room[nt:] < 0).sum())
        if infeasible_train or infeasible_validation:
            trials.append({'anchor': anchor, 'eligible': False, 'reason': 'Negative native energy remainder',
                'infeasible_training_rows': infeasible_train, 'infeasible_validation_rows': infeasible_validation})
            continue
        sign = 1. if float(RH[:nt, anchor].mean()) >= 0 else -1.
        anchor_target = sign * room.sqrt()
        anchor_rhs = V.T @ (ZN.T @ (anchor_target[:nt, None] / hr[:nt]) / nt)
        for ridge in ridges:
            fit = (V @ (anchor_rhs / (ev[:, None] + ridge))) / sdn[:, None]
            fitted = XN[:nt] @ fit
            scale = float(((anchor_target[:nt, None] / hr[:nt]).square().sum() / fitted.square().sum()).sqrt())
            W = W0.clone(); W[anchor, native] = fit[:, 0] * scale
            W = W.float()
            assert torch.equal(W[:, units], programme)
            q = quality(W, gain)
            count = bm['changed_scalars'] - bm['per_tensor'][dn] - bm['per_tensor'][pn]
            count += int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            count += int((parameter.view(torch.int32) != h[pn].view(torch.int32)).sum())
            row = {'anchor': anchor, 'anchor_ridge': ridge, 'anchor_sign': sign,
                'anchor_gain': float(gain[anchor]), 'analytic_energy_trace_scale': scale,
                **q, 'changed_scalars_if_applied_alone': count, 'maximum_down_weight': float(W.abs().max()),
                'infeasible_training_rows': 0, 'infeasible_validation_rows': 0}
            row['eligible'] = bool(count <= 5016268 and W.isfinite().all() and W.abs().max() <= 1e6
                and q['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_native_branch_rms'] < best[0]):
                best = q['validation_native_branch_rms'], W.clone(), parameter.clone(), row
    record = {'baseline': baseline, 'donor_count': len(donors), 'donor_ridge': .001,
        'strong_rows': strong.tolist(), 'existing_dense_rows': pool.tolist(),
        'zero_gain_rows': zero_gain.nonzero().flatten().tolist(),
        'strong_ridge_trials': ridges, 'strong_trials': strong_trials,
        'selected_strong_ridge': strong_best[2], 'anchor_coordinate_candidates': anchors.tolist(),
        'eligible_quiet_coordinate_count': int(torch.isfinite(importance).sum()),
        'anchor_ridge_trials': ridges, 'fixed_programme_gain_rows': fixed_gain_rows.nonzero().flatten().tolist(),
        'trials': trials, 'selected': None if best is None else best[-1], 'accepted_local_candidate': best is not None}
    reports[str(layer)] = record
    if best is not None: selected[dn], selected[pn] = best[1], best[2]
    print(json.dumps({'layer': layer, 'baseline': baseline, 'selected': record['selected']}), flush=True)
    del FH, F, before, teacher, trms, RH, hr, target, XN, ZN, X, Z, C, ev, V, rhs, W0, R0, strong_best

combined_count = bm['changed_scalars']
for name, value in selected.items():
    combined_count -= bm['per_tensor'][name]
    combined_count += int((value.view(torch.int32) != h[name].view(torch.int32)).sum())
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'training_windows': 192, 'validation_windows': 64, 'layers': reports,
    'selected_tensor_names': list(selected), 'combined_changed_scalars': combined_count,
    'target': 'Original HOST native branch and raw-energy identity at exact parent calibration inputs',
    'ordinary_gains': 'HOST, except every row touched by a fixed programme output',
    'format_caveat': 'No exact current-parent format capture after O3 repair; any candidate requires official arithmetic scoring',
    'programme_G_Up_Down_and_their_output_gains_fixed': True,
    'energy_policy': 'Reject a coordinate if any training or validation energy remainder is negative; no clipping',
    'no_answers_gradients_native_activation_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ap, lp, cp, hp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a188v1_early_native_energy_anchors.json'
atomic_json(out, report)
if selected and combined_count <= 5016268:
    atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected': list(selected), 'combined_changed_scalars': combined_count,
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
