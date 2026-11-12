"""Independent single-layer paired-HOST state repairs on existing edit support.

Every candidate starts from the same certified parent and original paired
HOST state targets. Parent input features and raw RMS stay fixed in every
solve. Candidate layers are alternatives, never composed into repeated
residual corrections; no native activation, gradient or answer target enters.
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
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
base = ROOT / meta['base']; assert sha(base) == meta['base_sha256']
ch = expand(h, base)
a, host, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, tp)]
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
assert torch.equal(a['inputs'], host['inputs'])
assert torch.equal(a['inputs'], top['inputs'][256:])
nt = 192 * 64
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
teacher_states = {}
for layer in range(18):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    if layer in meta['layers']:
        teacher_states[layer] = state.reshape(-1, 640).double()
assert torch.equal(state, host['captures']['finalraw'])
del state

reports, selected = {}, {}
for layer in meta['layers']:
    if layer == 9:
        reports['9'] = {'unmodified_observation_only': True, 'accepted_local_candidate': False,
                        'diagnostic_metrics': certificate['layers']['9']}
        continue
    pre = f'model.layers.{layer}.'
    dn = pre + 'mlp.down_proj.weight'; pn = pre + 'post_feedforward_layernorm.weight'
    nn = pre + 'pre_feedforward_layernorm.weight'
    assert torch.equal(ch.get(nn, h[nn]), h[nn])
    units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    F = FH.clone(); F[:, units] = 0
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    W0 = ch[dn].double(); gamma = 1 + ch[pn].double()
    original_post = norm(FH @ h[dn].double().T) * (1 + h[pn].double())
    raw0 = F @ W0.T
    rms0 = (raw0.square().mean(1, keepdim=True) + 1e-6).sqrt()
    current_post = raw0 / rms0 * gamma
    teacher = teacher_states[layer]
    teacher_rms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    # The support is fixed by the original parent, not by a fitted candidate.
    support = (ch[dn].view(torch.int32) != h[dn].view(torch.int32)).clone()
    support[:, units] = False
    support[gamma.abs() < 1e-8] = False
    patterns, groups = torch.unique(support, dim=0, return_inverse=True)
    corrections = [torch.zeros_like(W0) for _ in ridges]
    group_report = []
    for group in range(len(patterns)):
        rows = (groups == group).nonzero().flatten()
        columns = patterns[group].nonzero().flatten()
        if not len(columns):
            continue
        X = F[:nt, columns] / rms0[:nt]
        sd = X.square().mean(0).sqrt().clamp_min(1e-15)
        Z = X / sd
        C = Z.T @ Z / nt
        ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
        target = (teacher[:nt, rows] - before[:nt, rows] - current_post[:nt, rows]) / gamma[rows]
        rhs = V.T @ (Z.T @ target / nt)
        for i, ridge in enumerate(ridges):
            fit = (V @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
            corrections[i][rows[:, None], columns[None]] = fit.T
        group_report.append({'rows': rows.tolist(), 'native_feature_count': len(columns)})
        del X, Z, C, ev, V, target, rhs, fit

    def quality(W):
        raw = F[nt:] @ W.double().T
        post = norm(raw) * gamma
        output = before[nt:] + post
        delta = output - teacher[nt:]
        branch_error = post - original_post[nt:]
        row = {
            'validation_same_input_HOST_branch_rms': float(branch_error.square().mean().sqrt()),
            'validation_same_input_HOST_branch_relative_rms': float(branch_error.square().mean().sqrt() / original_post[nt:].square().mean().sqrt()),
            'validation_paired_HOST_state_relative_rms': float((delta / teacher_rms[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
            'maximum_relative_native_raw_rms_change': float(((raw.square().mean(1, keepdim=True) + 1e-6).sqrt() / rms0[nt:] - 1).abs().max())}
        if layer == 17:
            ids = torch.cat([top['topk_ids'][256 + 192:], a['topk_ids'][192:]], -1).reshape(-1, 16).long()
            rivals = h['model.embed_tokens.weight'][ids].double() * (1 + h['model.norm.weight'].double())
            scores = torch.einsum('nd,nkd->nk', output, rivals)
            winners = ids.gather(1, scores.argmax(1, keepdim=True)).flatten()
            row['validation_restricted_matches'] = int((winners == ids[:, 0]).sum())
        return row, output

    baseline, baseline_state = quality(W0)
    trials, best = [], None
    for ridge, correction in zip(ridges, corrections):
        for strength in strengths:
            W = ch[dn].clone()
            W[support] = (W0 + strength * correction)[support].float()
            assert torch.equal(W[:, units], ch[dn][:, units])
            assert torch.equal(W[~support], ch[dn][~support])
            q, output = quality(W)
            changed = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            row = {'native_ridge': ridge, 'correction_strength': strength, **q,
                   'maximum_down_weight': float(W.abs().max()), 'changed_down_scalars': changed}
            row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
                and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
            if layer == 17:
                row['eligible'] &= q['validation_restricted_matches'] > baseline['validation_restricted_matches']
                key = (q['validation_restricted_matches'], -q['validation_paired_HOST_state_relative_rms'])
            else:
                key = (-q['validation_paired_HOST_state_relative_rms'],)
            trials.append(row)
            if row['eligible'] and (best is None or key > best[0]):
                best = key, W.clone(), output.clone(), row
    record = {'baseline': baseline, 'support_groups': group_report,
              'existing_allowed_native_edits': int(support.sum()),
              'trials': trials, 'selected': None if best is None else best[-1]}
    if best is not None and layer == 17:
        E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
        outputs = (norm(torch.cat([baseline_state, best[2]])) * fg).float()
        maxima = torch.full((len(outputs),), -torch.inf)
        winners = torch.full((len(outputs),), -1, dtype=torch.long)
        for start in range(0, len(E), 4096):
            scores = outputs @ E[start:start + 4096].T
            values, indices = scores.max(1); better = values > maxima
            maxima[better] = values[better]; winners[better] = start + indices[better]
        truth = top['topk_ids'][256 + 192:, :, 0].reshape(-1)
        n = len(truth)
        full = {'baseline_matches': int((winners[:n] == truth).sum()),
                'selected_matches': int((winners[n:] == truth).sum()), 'tokens': n}
        recorded_matches = int((a['topk_ids'][192:, :, 0].reshape(-1) == truth).sum())
        full['recorded_parent_matches'] = recorded_matches
        assert abs(full['baseline_matches'] - recorded_matches) <= 2
        record['full_vocabulary_validation'] = full
        if full['selected_matches'] <= full['baseline_matches']:
            record['rejected_by_full_vocabulary_check'] = True
            best = None
        del outputs, maxima, winners, scores
    record['accepted_local_candidate'] = best is not None
    if best is not None:
        selected[dn] = best[1]
    reports[str(layer)] = record
    print(json.dumps({'layer': layer, 'baseline': baseline, 'selected': record['selected'],
                      'accepted_local_candidate': record['accepted_local_candidate']}), flush=True)
    del FH, F, before, W0, original_post, raw0, rms0, current_post, teacher, teacher_rms
    del support, patterns, groups, corrections, correction, W, output, baseline_state, best

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'native_ridge_trials': ridges, 'correction_strength_trials': strengths,
          'training_windows': 192, 'validation_windows': 64,
          'target': 'Original paired HOST state after this layer minus the fixed parent pre-MLP residual; one-shot fixed-parent normalizer',
          'single_layer_alternatives_never_composed': True,
          'support': 'Only already edited native Down entries; programme columns and gains unchanged',
          'format_caveat': 'No exact current-parent format capture is available after its O3 repair; selected edits require official arithmetic scoring',
          'layers': reports, 'selected_tensor_names': list(selected),
          'no_answers_gradients_native_activation_replay_or_iterative_refit': True,
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ap, lp, cp, hp, tp]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a180v1_paired_native_mlp_repair.json'
atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected_layers': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
