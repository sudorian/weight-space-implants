"""One-shot early state repair with a norm-orthogonal geometric target.

Project the paired original-HOST state discrepancy into raw branch directions
orthogonal to the observed current raw vector. This is a closed-form geometric
projection with five regularizers, not gradients or iterative fitting. Each
independent ridge solve uses the same frozen native-map prior and original
HOST target; no A151 fitted weights or any newly fitted residual are reused.
Only native Down columns in one layer change in any saved checkpoint.
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
states = {mode: expand(h, path) for mode, path in bases.items()}
manifests = {mode: json.loads(path.with_suffix('.manifest.json').read_text()) for mode, path in bases.items()}
reference = expand(h, capture_base)
w, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp)]
layout = json.loads(lp.read_text())
assert w['checkpoint_sha256'] == sha(capture_base)
assert torch.equal(w['inputs'][:256], host['inputs']) and host['report']['final_state_bit_equal']
for layer in range(8):
    for name in h:
        if name.startswith(f'model.layers.{layer}.'):
            for ch in states.values():
                assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
ch = states['leader']; nt = 192 * 64
regularizers = [1e-6, 1e-4, .01, 1., 100.]
ridges = [.001, .01, .1, 1., 10.]
report = {'training_windows': 192, 'validation_windows': 64, 'format_windows': 128,
          'projection_regularizer_trials': regularizers, 'ridge_trials': ridges,
          'donor_count_rule': '[256,512,640,1728,all surviving native features]',
          'projection': 'Minimum diagonal output-error plus raw-step penalty, constrained orthogonal to current raw unit vector',
          'fixed_prior': 'Unchanged early native maps shared by A142v2/A136; no A151 candidate reused',
          'reference_normalizer': 'Current recorded raw MLP RMS, fixed before all fits',
          'target': 'Paired original HOST states on unlabeled calibration text',
          'minimum_state_and_direction_improvement_fraction': .05,
          'format_receiver_limit': 100., 'format_source_limit_layers1_to3': 1., 'format_source_limit_layers4_to7': 100.,
          'selection': 'Largest absolute validation relative-state-RMS reduction among eligible independent single-layer maps',
          'no_answers_gradients_activation_replay_or_iterative_residual_refit': True, 'layers': {}}
report_path = ROOT / 'artifacts/a152v1_norm_orthogonal_state_repair.json'
best = {'leader': None, 'compact': None}

for layer in range(1, 8):
    prefix = f'model.layers.{layer}.'
    dn, pn = prefix + 'mlp.down_proj.weight', prefix + 'post_feedforward_layernorm.weight'
    Wcurrent = ch.get(dn, h[dn]).double(); gamma = 1 + ch.get(pn, h[pn]).double()
    units = torch.tensor(layout['programme_units'][str(layer)], dtype=torch.long)
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
    F = w['captures'][f'word.{layer}.features'].double().reshape(-1, 2048)
    assert F[:, units].abs().max() == 0
    before = w['captures'][f'word.{layer}.raw'].double().reshape(-1, 640)
    teacher = host['states'][f'{layer}.out'].double().reshape(-1, 640)
    raw = F @ Wcurrent.T
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt(); raw_n = raw / rms
    ref = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    current = before + raw_n * gamma
    def quality(r):
        output = before[nt:] + r / (r.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
        error = (output - teacher[nt:]) / ref[nt:]
        direction = output / (output.square().mean(1, keepdim=True) + 1e-6).sqrt() - teacher[nt:] / ref[nt:]
        return {'relative_hidden_state_rms': float(error.square().mean().sqrt()),
                'hidden_direction_rms': float(direction.square().mean().sqrt()),
                'relative_token_error_p99': float(error.square().mean(1).sqrt().quantile(.99))}
    baseline = quality(raw[nt:])
    ordinary = (gamma.abs() < 1e5) & (gamma.abs() > 1e-12)
    regularizer_scale = float(gamma[ordinary].square().median())
    E = teacher - current
    geometries = []; geometry = None
    for value in regularizers:
        inverse = 1 / (gamma.square() + value * regularizer_scale)
        first = E * gamma * inverse
        constraint = raw_n * inverse
        multiplier = (raw_n * first).sum(1, keepdim=True) / (raw_n * constraint).sum(1, keepdim=True).clamp_min(1e-30)
        step = first - constraint * multiplier
        orthogonality = float((raw_n * step).sum(1).abs().max())
        ideal = quality((raw_n[nt:] + step[nt:]) * rms[nt:])
        row = {'projection_regularizer': value, 'regularizer_scale': regularizer_scale,
               'maximum_raw_orthogonality_error': orthogonality,
               'maximum_relative_raw_norm_change': float((((raw_n + step).square().mean(1) + 1e-6 / rms[:, 0].square()).sqrt() - 1).abs().max()),
               'ideal': ideal}
        row['eligible'] = bool(step.isfinite().all() and ideal['relative_hidden_state_rms'] < .5 * baseline['relative_hidden_state_rms'])
        geometries.append(row)
        if row['eligible'] and (geometry is None or ideal['relative_hidden_state_rms'] < geometry[0]):
            geometry = ideal['relative_hidden_state_rms'], step.clone(), row
    layer_report = {'baseline': baseline, 'geometry_trials': geometries, 'trials': []}
    report['layers'][str(layer)] = layer_report
    print(json.dumps({'layer': layer, 'baseline': baseline, 'geometry': geometries}), flush=True)
    if geometry is None:
        continue
    _, target_step, selected_geometry = geometry
    layer_report['selected_geometry'] = selected_geometry
    Fn = F[:nt] / rms[:nt] / ref[:nt]
    standard = Fn.square().mean(0).sqrt().clamp_min(1e-12)
    Z = Fn[:, native] / standard[native]
    cross = Z.T @ (target_step[:nt] / ref[:nt]) / nt
    priority = (cross * gamma).square().sum(1)
    changed_native = (ch[dn].view(torch.int32) != h[dn].view(torch.int32))[ordinary].any(0)
    ranked = priority.argsort(descending=True)
    # Prefer existing edited columns; this is a fixed allocation rule, not
    # a second fitted model or extra scale grid.
    rank_existing = ranked[changed_native[native[ranked]]]
    rank_new = ranked[~changed_native[native[ranked]]]
    ranking = torch.cat([rank_existing, rank_new])
    columns_all = native[ranking]
    Z = Z[:, ranking]; covariance = Z.T @ Z / nt; rhs = cross[ranking]
    counts = [256, 512, 640, 1728, len(native)]
    layer_report['donor_counts'] = counts
    layer_report['existing_edited_native_columns'] = len(rank_existing)
    del Fn, Z, cross, target_step, step, first, constraint, E
    PF = w['captures'][f'format.{layer}.features'].double().reshape(-1, 2048)
    PR = PF @ Wcurrent.T
    post0 = PR / (PR.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
    for count in counts:
        columns = columns_all[:count]
        C = covariance[:count, :count]
        ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
        projected = V.T @ rhs[:count]
        for ridge in ridges:
            correction = (V @ (projected / (ev[:, None] + ridge))) / standard[columns, None]
            W = Wcurrent.clone(); W[:, columns] += correction.T; W = W.float()
            assert torch.equal(W[:, units], Wcurrent[:, units].float())
            q = quality(F[nt:] @ W.double().T)
            fr = PF @ W.double().T
            post = fr / (fr.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
            change = (post - post0).abs().reshape(128, 14, 640)
            source_change = float(change[:, :8].max()); receiver_change = float(change[:, 8:].max())
            changed_tensor = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            row = {'layer': layer, 'donor_count': count, 'ridge': ridge, **q,
                   'maximum_format_source_post_change': source_change,
                   'maximum_format_receiver_post_change': receiver_change,
                   'changed_down_scalars': changed_tensor, 'maximum_down_weight': float(W.abs().max()), 'allocation': {}}
            good = bool(W.isfinite().all() and W.abs().max() <= 1e6 and source_change < (1. if layer <= 3 else 100.) and
                        receiver_change < 100. and q['relative_token_error_p99'] < 1.1 * baseline['relative_token_error_p99'] and
                        q['relative_hidden_state_rms'] < .95 * baseline['relative_hidden_state_rms'] and
                        q['hidden_direction_rms'] < .95 * baseline['hidden_direction_rms'])
            for mode, bm in manifests.items():
                changed = bm['changed_scalars'] - bm['per_tensor'].get(dn, 0) + changed_tensor
                eligible = good and changed <= int(.05 * ELIGIBLE)
                row['allocation'][mode] = {'changed_scalars': changed, 'eligible': eligible}
                key = baseline['relative_hidden_state_rms'] - q['relative_hidden_state_rms']
                if eligible and (best[mode] is None or key > best[mode][0]):
                    best[mode] = key, dn, W.clone(), units.clone(), layer, copy.deepcopy(row)
            layer_report['trials'].append(row)
            print(json.dumps(row), flush=True)
        atomic_json(report_path, report)
    print(json.dumps({'completed_layer': layer, 'baseline': baseline,
                      'best_trial': min(layer_report['trials'], key=lambda r: r['relative_hidden_state_rms'])}), flush=True)

report['selected'] = {mode: None if item is None else item[5] for mode, item in best.items()}
report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['static_caveat'] = 'Exact-input local state/format checks do not establish complete downstream text predictions or arithmetic accuracy.'
atomic_json(report_path, report)
for mode, item in best.items():
    if item is None:
        print(json.dumps({'mode': mode, 'checkpoint': None, 'reason': 'No eligible five-percent independent repair'}), flush=True)
        continue
    base = bases[mode]; changed = {name: value.clone() for name, value in states[mode].items()}
    c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
    _, dn, W, units, layer, selected = item
    changed[dn] = W
    local_report = {**report, 'selected_mode': mode, 'selected_layer': layer, 'selected_trial': selected}
    c['norm_orthogonal_state_repair_a152'] = local_report
    out = ROOT / f'experiments/a152v1_norm_orthogonal_{mode}/delta.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out.parent / 'norm_orthogonal_state_repair.json', local_report)
    atomic_torch(out.parent / 'circuit.pt', c)
    bm = manifests[mode]; categories = copy.deepcopy(bm['categories'])
    mask = W.view(torch.int32) != h[dn].view(torch.int32)
    categories[dn] = {'arithmetic_programme': int(mask[:, units].sum()),
                      'native_text_state_repair': int(mask.sum() - mask[:, units].sum())}
    inputs = [wp, hp, lp, capture_base]
    for path in bases.values():
        inputs.extend([path, path.with_suffix('.manifest.json'), path.parent / 'circuit.pt'])
    save(out, h, changed,
         sources=[Path(__file__), ROOT / 'experiments/derive_a150_recorded_host_states_v1.py'] + [ROOT / s for s in bm['sources']],
         inputs=inputs, arguments={'allocation': mode}, description=__doc__, extra=local_report, categories=categories)
