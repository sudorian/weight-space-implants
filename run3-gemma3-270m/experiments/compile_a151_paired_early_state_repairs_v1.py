"""Independent original-HOST state repairs at exact current early inputs.

Each layer/count/ridge solve starts from HOST Down with fixed strong-writer
accommodation and unchanged arithmetic columns. The target is derived once
from paired unlabeled original HOST states, a fixed observed normalizer and
the smallest eligible weak-anchor geometry. Two existing allocation layouts
are considered independently. No gradients, activation/attention replay,
whole-problem targets, or iterative residual fitting are used.
"""
import copy
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha, ELIGIBLE

h = host_state()
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
hp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
lp = ROOT / 'artifacts/a150v1_early_native_basis_layout.json'
capture_base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
bases = {'leader': ROOT / 'experiments/a142v2_exploratory_projected/delta.pt',
         'compact': ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'}
w, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp)]
layout = json.loads(lp.read_text())
reference = expand(h, capture_base)
assert w['checkpoint_sha256'] == sha(capture_base)
assert torch.equal(w['inputs'][:256], host['inputs']) and host['report']['final_state_bit_equal']
nt = 192 * 64
anchor_counts = [1, 4, 16, 64, 128]
ridges = [.001, .01, .1, 1., 10.]
report = {'training_windows': 192, 'validation_windows': 64, 'format_windows': 128,
          'anchor_count_trials': anchor_counts, 'ridge_trials': ridges,
          'donor_count_rule': '[256,512,640,1728,all surviving native features]',
          'HOST_ray_scale_interval': [.5, 2.], 'minimum_state_and_direction_improvement_fraction': .05,
          'format_receiver_maximum_post_change': 100.,
          'format_source_maximum_post_change_layers1_to3': 1.,
          'format_source_maximum_post_change_layers4_to8': 100.,
          'selection': 'Largest absolute validation relative-state-RMS reduction among eligible independent single-layer repairs',
          'all_programme_columns_G_Up_and_gains_fixed': True,
          'no_answers_gradients_activation_replay_or_iterative_residual_refit': True,
          'modes': {}}
report_path = ROOT / 'artifacts/a151v1_paired_early_state_repairs.json'
selected_by_mode = {}

for mode, base in bases.items():
    ch = expand(h, base)
    bm = json.loads(base.with_suffix('.manifest.json').read_text())
    # Exact observed input to every assessed MLP. The compact alternative
    # differs only in MLP8 Down/gains, after its recorded input/features.
    for layer in range(9):
        for name in h:
            if not name.startswith(f'model.layers.{layer}.'):
                continue
            if layer == 8 and name.endswith(('mlp.down_proj.weight', 'post_feedforward_layernorm.weight')):
                continue
            assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
    mode_report = {'base': str(base.relative_to(ROOT)), 'base_changed_scalars': bm['changed_scalars'], 'layers': {}}
    best = None
    for layer in range(1, 9):
        prefix = f'model.layers.{layer}.'
        dn, pn = prefix + 'mlp.down_proj.weight', prefix + 'post_feedforward_layernorm.weight'
        Wcurrent = ch.get(dn, h[dn]).double(); gamma = 1 + ch.get(pn, h[pn]).double()
        gh = 1 + h[pn].double()
        units = torch.tensor(layout['programme_units'][str(layer)], dtype=torch.long)
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
        F = w['captures'][f'word.{layer}.features'].double().reshape(-1, 2048)
        assert F[:, units].abs().max() == 0
        before = w['captures'][f'word.{layer}.raw'].double().reshape(-1, 640)
        teacher = host['states'][f'{layer}.out'].double().reshape(-1, 640)
        raw = F @ Wcurrent.T
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt(); raw_n = raw / rms
        ref = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
        def quality(r):
            output = before[nt:] + r / (r.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
            error = (output - teacher[nt:]) / ref[nt:]
            direction = output / (output.square().mean(1, keepdim=True) + 1e-6).sqrt() - teacher[nt:] / ref[nt:]
            return {'relative_hidden_state_rms': float(error.square().mean().sqrt()),
                    'hidden_direction_rms': float(direction.square().mean().sqrt()),
                    'relative_token_error_p99': float(error.square().mean(1).sqrt().quantile(.99))}
        baseline = quality(raw[nt:])
        zero = gamma.abs() < 1e-12
        divisor = torch.where(zero, torch.ones_like(gamma), gamma)
        order = gamma.abs().argsort()
        geometries = []; choice = None
        for count in anchor_counts:
            anchors = order[:max(count, int(zero.sum()))]
            A, B = teacher / divisor, before / divisor
            A[:, anchors], B[:, anchors] = 0, 0
            radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
            aa = A.square().sum(1).clamp_min(1e-30)
            centre = (A * B).sum(1) / aa
            minimum = (B - centre[:, None] * A).square().sum(1)
            half = ((radius - minimum).clamp_min(0) / aa).sqrt()
            lo = torch.maximum(centre - half, torch.full_like(centre, .5))
            hi = torch.minimum(centre + half, torch.full_like(centre, 2.))
            feasible = (minimum <= radius) & (lo <= hi)
            scale = torch.minimum(torch.maximum(torch.ones_like(lo), lo), hi)
            target = A * scale[:, None] - B
            room = (radius - target.square().sum(1)).clamp_min(0)
            direction = raw_n[:, anchors].clone()
            empty = direction.norm(dim=1) < 1e-30; direction[empty, 0] = 1
            direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
            target[:, anchors] = direction * room.sqrt()[:, None]
            target[~feasible] = raw_n[~feasible]
            ideal = quality(target[nt:] * rms[nt:])
            row = {'requested_anchor_count': count, 'actual_anchor_count': len(anchors),
                   'anchors': anchors.tolist(), 'feasible_training_fraction': float(feasible[:nt].double().mean()),
                   'infeasible_rows': int((~feasible).sum()), 'ideal': ideal}
            row['eligible'] = (row['feasible_training_fraction'] >= .95 and
                               ideal['hidden_direction_rms'] < .25 * baseline['hidden_direction_rms'] and
                               ideal['relative_hidden_state_rms'] < .5 * baseline['relative_hidden_state_rms'])
            geometries.append(row)
            if row['eligible'] and choice is None:
                choice = target.clone(), feasible.clone(), row
        layer_report = {'baseline': baseline, 'geometry': geometries, 'trials': []}
        mode_report['layers'][str(layer)] = layer_report
        if choice is None:
            continue
        target, feasible, selected_geometry = choice
        layer_report['selected_geometry'] = selected_geometry
        used = torch.arange(len(F)) < nt; used &= feasible
        Worigin = h[dn].double().clone()
        strong = gamma.abs() > 1e5
        Worigin[strong] *= (gh[strong] / gamma[strong])[:, None]
        Worigin[:, units] = Wcurrent[:, units]
        origin_raw = F @ Worigin.T
        Fn = F / rms / ref
        standard = Fn[used].square().mean(0).sqrt().clamp_min(1e-12)
        Z = Fn[used][:, native] / standard[native]
        Y = (target[used] - origin_raw[used] / rms[used]) / ref[used]
        cross = Z.T @ Y / len(Z)
        priority = (cross * gamma).square().sum(1)
        ranking = priority.argsort(descending=True)
        columns_all = native[ranking]
        Z = Z[:, ranking]
        covariance = Z.T @ Z / len(Z)
        rhs = cross[ranking]
        counts = [256, 512, 640, 1728, len(native)]
        layer_report['donor_counts'] = counts
        layer_report['training_rows'] = int(used.sum())
        del Z, Y, Fn, cross, target, origin_raw, A, B, direction
        PF = w['captures'][f'format.{layer}.features'].double().reshape(-1, 2048)
        PR = PF @ Wcurrent.T
        post0 = PR / (PR.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
        other_count = bm['changed_scalars'] - bm['per_tensor'].get(dn, 0)
        for count in counts:
            columns = columns_all[:count]
            C = covariance[:count, :count]
            ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
            projected = V.T @ rhs[:count]
            for ridge in ridges:
                correction = (V @ (projected / (ev[:, None] + ridge))) / standard[columns, None]
                W = Worigin.clone(); W[:, columns] += correction.T; W = W.float()
                assert torch.equal(W[:, units], Wcurrent[:, units].float())
                q = quality(F[nt:] @ W.double().T)
                fr = PF @ W.double().T
                post = fr / (fr.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
                change = (post - post0).abs().reshape(128, 14, 640)
                source_change = float(change[:, :8].max()); receiver_change = float(change[:, 8:].max())
                changed = other_count + int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
                row = {'layer': layer, 'donor_count': count, 'ridge': ridge, **q,
                       'maximum_format_source_post_change': source_change,
                       'maximum_format_receiver_post_change': receiver_change,
                       'changed_scalars': changed, 'maximum_down_weight': float(W.abs().max())}
                row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6 and changed <= int(.05 * ELIGIBLE) and
                    source_change < (1. if layer <= 3 else 100.) and receiver_change < 100. and
                    q['relative_token_error_p99'] < 1.1 * baseline['relative_token_error_p99'] and
                    q['relative_hidden_state_rms'] < .95 * baseline['relative_hidden_state_rms'] and
                    q['hidden_direction_rms'] < .95 * baseline['hidden_direction_rms'])
                layer_report['trials'].append(row)
                print(json.dumps({'mode': mode, **row}), flush=True)
                key = baseline['relative_hidden_state_rms'] - q['relative_hidden_state_rms']
                if row['eligible'] and (best is None or key > best[0]):
                    best = key, dn, W.clone(), units.clone(), copy.deepcopy(row), layer
            report['modes'][mode] = mode_report
            atomic_json(report_path, report)
        print(json.dumps({'mode': mode, 'completed_layer': layer, 'baseline': baseline,
                          'best_layer_trial': min(layer_report['trials'], key=lambda r: r['relative_hidden_state_rms'])}), flush=True)
    mode_report['selected'] = None if best is None else best[4]
    report['modes'][mode] = mode_report
    selected_by_mode[mode] = best
    atomic_json(report_path, report)

report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['static_caveat'] = 'Local exact-input branch/readout checks do not establish downstream text predictions or complete arithmetic accuracy.'
atomic_json(report_path, report)
for mode, best in selected_by_mode.items():
    if best is None:
        print(json.dumps({'mode': mode, 'checkpoint': None, 'reason': 'No eligible five-percent state-and-direction improvement'}), flush=True)
        continue
    base = bases[mode]; ch = expand(h, base)
    c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
    _, dn, W, units, selected, layer = best
    ch[dn] = W
    local_report = {**report, 'selected_mode': mode, 'selected_layer': layer, 'selected': selected}
    c['paired_early_state_repair_a151'] = local_report
    out = ROOT / f'experiments/a151v1_early_state_{mode}/delta.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out.parent / 'paired_early_state_repair.json', local_report)
    atomic_torch(out.parent / 'circuit.pt', c)
    bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
    changed = W.view(torch.int32) != h[dn].view(torch.int32)
    categories[dn] = {'arithmetic_programme': int(changed[:, units].sum()),
                      'native_text_state_repair': int(changed.sum() - changed[:, units].sum())}
    save(out, h, ch,
         sources=[Path(__file__), ROOT / 'experiments/derive_a150_recorded_host_states_v1.py'] + [ROOT / s for s in bm['sources']],
         inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, lp, capture_base],
         arguments={'allocation': mode}, description=__doc__, extra=local_report, categories=categories)
