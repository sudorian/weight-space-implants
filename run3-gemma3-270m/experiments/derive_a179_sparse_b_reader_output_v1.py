"""Sparse B-reader output repair from uniquely recovered recorded head values.

Only linear inversion of the observed branch output, recorded-state algebra,
RMS arithmetic and closed-form original-HOST text reconstruction are used.
The entire five-function arithmetic frame is protected before normalization.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a176v1_sparse_frame_null_output/delta.pt'
ch = expand(h, base); cp = base.parent / 'circuit.pt'
c = torch.load(cp, weights_only=True)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
sp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
rp = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
w, e, host, hs = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp, sp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(rp)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'][:256], host['inputs'])
reference = expand(h, rp)
for name in h:
    if any(name.startswith(f'model.layers.{i}.') for i in [0, 1]) or (
        name.startswith('model.layers.2.') and ('.self_attn.' in name or name.endswith('input_layernorm.weight') or name.endswith('post_attention_layernorm.weight'))):
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
pre = 'model.layers.2.'
on = pre + 'self_attn.o_proj.weight'; vn = pre + 'self_attn.v_proj.weight'
assert torch.equal(ch.get(on, h[on]), h[on]) and torch.equal(ch.get(vn, h[vn]), h[vn])
O0 = h[on].double(); O = O0.reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
frame = c['native_b_stage']['reader']['frame'].double(); assert frame.shape == (640, 5)
units = c['native_b_stage']['units'].long()
G = ch[pre + 'mlp.gate_proj.weight'][units].double()

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def recover(X, observed, label, value_weight=None):
    # This is the full-rank inverse method independently checked in A171.
    # Its right-hand side is a captured output, never synthesized attention.
    value_weight = h[vn] if value_weight is None else value_weight
    values = (X @ value_weight.T).double()
    heads, ranks, ratios = [], [], []
    squared_error = 0.; squared_reference = 0.; future_max = 0.
    minimum = 1.; sum_error = 0.
    future = torch.ones(X.shape[1], X.shape[1], dtype=torch.bool).triu(1)
    for i in range(len(X)):
        matrix = torch.einsum('ohv,sv->ohs', O, values[i]).reshape(640, -1)
        result = torch.linalg.lstsq(matrix, observed[i].double().T, rcond=1e-10, driver='gelsd')
        coef = result.solution.T.reshape(X.shape[1], 4, X.shape[1]).permute(1, 0, 2)
        head = coef @ values[i]
        reproduced = torch.einsum('htv,ohv->to', head, O)
        squared_error += float((reproduced - observed[i]).square().sum())
        squared_reference += float(observed[i].double().square().sum())
        ranks.append(int(result.rank))
        ratios.append(float(result.singular_values[-1] / result.singular_values[0]))
        future_max = max(future_max, float(coef[:, future].abs().max()))
        minimum = min(minimum, float(coef.min()))
        sum_error = max(sum_error, float((coef.sum(-1) - 1).abs().max()))
        heads.append(head.permute(1, 0, 2).reshape(X.shape[1], 1024).float())
    record = {'windows': len(X), 'tokens_per_window': X.shape[1],
        'relative_raw_reconstruction_error': (squared_error / squared_reference) ** .5,
        'minimum_matrix_rank': min(ranks), 'full_column_count': 4 * X.shape[1],
        'minimum_singular_ratio': min(ratios), 'maximum_future_coefficient': future_max,
        'minimum_coefficient': minimum, 'maximum_coefficient_sum_error': sum_error}
    assert record['minimum_matrix_rank'] == record['full_column_count']
    assert record['relative_raw_reconstruction_error'] < 1e-5
    assert future_max < .002 and minimum > -.002 and sum_error < .002
    print(json.dumps({'recovery': label, **record}), flush=True)
    return torch.stack(heads).double(), record

known_path = ROOT / 'artifacts/a126v1_mixed_audit.pt'
known_base = ROOT / 'experiments/a126v1_receiver_free_rank_one_b/delta.pt'
known = torch.load(known_path, weights_only=True, mmap=True)
assert known['checkpoint_sha256'] == sha(known_base)
known_state = expand(h, known_base)
assert torch.equal(known_state.get(on, h[on]), h[on])
known_x = known['captures']['word.2.input'][:8]
known_heads = known['captures']['word.2.heads'][:8]
assert known_heads.shape == (8, 64, 1024)
known_raw = (known_heads.double() @ O0.T).float()
recovered, independent_check = recover(known_x, known_raw, 'known_heads', known_state.get(vn, h[vn]))
independent_check['relative_known_head_error'] = float((recovered - known_heads).norm() / known_heads.double().norm())
assert independent_check['relative_known_head_error'] < 1e-4
print(json.dumps({'independent_check': independent_check}), flush=True)
del known_state, known, known_x, known_heads, known_raw, recovered

heads_by_tag, before_by_tag, recoveries = {}, {}, {}
for tag in ['word', 'format']:
    before = w['captures'][f'{tag}.1.raw'] + fnorm(e['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    reproduced = before + fnorm(e['captures'][f'{tag}.2.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
    assert torch.equal(reproduced, w['captures'][f'{tag}.2.raw'])
    X = fnorm(before) * (1 + h[pre + 'input_layernorm.weight'])
    heads_by_tag[tag], recoveries[tag] = recover(X, e['captures'][f'{tag}.2.attention.raw'], tag)
    before_by_tag[tag] = before.double()
heads, fheads = heads_by_tag['word'], heads_by_tag['format']
word_before, format_before = before_by_tag['word'], before_by_tag['format']
raw = e['captures']['word.2.attention.raw'].double()
hr = host['captures']['2.attention.raw'].double()
assert (heads @ O0.T - raw).norm() / raw.norm() < 1e-5
teacher = hs['states']['1.out'].double() + norm(hr) * gamma
rms = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
target = (norm(hr) - norm(raw)) * gamma
nt = 192; nr = nt * 64
X = (heads / rms).reshape(-1, 1024); Y = target.reshape(-1, 640)
sd = X[:nr].square().mean(0).sqrt().clamp_min(1e-12)
Z = X[:nr] / sd; C = Z.T @ Z / nr; rhs = Z.T @ Y[:nr] / nr
priority = Y[:nr].square().mean(0); priority[gamma.abs() < .05] = -torch.inf
rows = priority.argsort(descending=True)[:128]
u, ss, _ = torch.linalg.svd(frame[rows], full_matrices=False)
rank = int((ss > ss[0] * 1e-10).sum()); B = u[:, :rank]
inverse_metric = 1 / (1 + .01 * float(gamma.square().median()) / gamma[rows].square())
Zm = inverse_metric[:, None] * B
projection = torch.eye(len(rows), dtype=torch.float64) - Zm @ torch.linalg.solve(B.T @ Zm, B.T)
ranking_gram = C.clone(); ranking_gram.diagonal().add_(.001)
precision = torch.linalg.inv(ranking_gram)
ranking_beta = precision @ (rhs[:, rows] @ projection.T)
feature_priority = ranking_beta.square().sum(1) / precision.diagonal().clamp_min(1e-15)
feature_order = feature_priority.argsort(descending=True)
counts = [16, 32, 64, 128, 256]
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
PR = fheads @ O0.T; PRMS = (PR.square().mean(-1, keepdim=True) + 1e-6).sqrt()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
tokens = w['inputs'][256:, :14]
expected = torch.stack([(tokens[:, pos, None] == ids).long().argmax(1) for pos in [7, 6, 5]], 1)

def quality(matrix):
    post = norm(heads[nt:] @ matrix.T) * gamma; hp = norm(hr[nt:]) * gamma
    after = word_before[nt:] + post; desired = teacher[nt:]
    scale = (desired.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    ppraw = fheads @ matrix.T; fstate = format_before + norm(ppraw) * gamma
    code = fstate[:, 8] @ frame
    decoded = 9 * code[:, :3] / code[:, 3:4]; de = (decoded - expected).abs()
    nrms = (ppraw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    return {'validation_hidden_relative_rms': float(((after - desired) / scale).square().mean().sqrt()),
        'validation_hidden_direction_rms': float((norm(after) - norm(desired)).square().mean().sqrt()),
        'validation_paired_HOST_branch_rms': float((post - hp).square().mean().sqrt()),
        'maximum_word_programme_gate': float(((norm(after) * pg) @ G.T).max()),
        'minimum_format_equals_gate': float(((norm(fstate) * pg) @ G.T)[:, 8].min()),
        'maximum_individual_source_digit_errors': de.amax(0).tolist(),
        'wrong_individual_source_digits': int((decoded.round() != expected).sum()),
        'minimum_format_reference': float(code[:, 3].min()),
        'maximum_format_relative_raw_rms_change': float((nrms / PRMS - 1).abs().max())}

baseline = quality(O0)
print(json.dumps({'baseline': baseline}), flush=True)
assert baseline['wrong_individual_source_digits'] == 0
bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials, best = [], None
for count in counts:
    columns = feature_order[:count]; gram = C[columns][:, columns]
    ce, cv = torch.linalg.eigh((gram + gram.T) / 2); ce = ce.clamp_min(0)
    pr = cv.T @ rhs[columns]
    for ridge in ridges:
        beta = (cv @ (pr / (ce[:, None] + ridge))) / sd[columns, None]
        correction = (projection @ beta[:, rows].T) / gamma[rows, None]
        for strength in strengths:
            matrix = O0.clone()
            matrix[rows[:, None], columns[None]] = (O0[rows[:, None], columns[None]] + strength * correction).float().double()
            q = quality(matrix)
            changed = bm['changed_scalars'] + int((matrix.float().view(torch.int32) != h[on].view(torch.int32)).sum())
            row = {'native_input_coordinate_count': count, 'output_row_count': len(rows),
                'native_ridge': ridge, 'correction_strength': strength, **q,
                'changed_scalars': changed, 'maximum_output_weight': float(matrix.abs().max()),
                'maximum_frame_null_error': float((frame.T @ (gamma[:, None] * (matrix - O0))).abs().max())}
            row['eligible'] = bool(changed <= 5016268 and matrix.isfinite().all() and matrix.abs().max() <= 1e6
                and q['maximum_word_programme_gate'] < -16 and q['minimum_format_equals_gate'] > 31.9
                and q['wrong_individual_source_digits'] == 0
                and q['minimum_format_reference'] > .5 * baseline['minimum_format_reference'] > 0
                and max(q['maximum_individual_source_digit_errors']) < min(.4, max(baseline['maximum_individual_source_digit_errors']) + .01)
                and q['maximum_format_relative_raw_rms_change'] < .02
                and q['validation_hidden_relative_rms'] < baseline['validation_hidden_relative_rms'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_paired_HOST_branch_rms'] < best[0]):
                best = q['validation_paired_HOST_branch_rms'], matrix.float(), row
        print(json.dumps({'native_input_coordinate_count': count, 'ridge': ridge, 'best': None if best is None else best[-1]}), flush=True)

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'independent_known_head_check': independent_check, 'recovered_head_checks': recoveries, 'baseline': baseline,
    'native_input_coordinate_count_trials': counts, 'output_row_count': len(rows),
    'native_ridge_trials': ridges, 'correction_strength_trials': strengths,
    'fixed_ranking_ridge': .001, 'fixed_raw_projection_penalty': .01,
    'training_windows': 192, 'validation_windows': 64, 'format_windows': 128,
    'source_readout': 'Nine times digit/9 field divided by common reference; individual source digits only',
    'protected_frame_rank': rank, 'protected_readout_function_count': 5,
    'native_input_coordinate_order': feature_order.tolist(), 'output_rows': rows.tolist(),
    'target': 'Original paired HOST normalized attention2 branch at fixed recorded current raw RMS',
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_model_replay_gradients_answers_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                'experiments/analyze_a171_recorded_attention_inverse_v1.py': sha(ROOT / 'experiments/analyze_a171_recorded_attention_inverse_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, wp, ep, hp, sp, rp, known_path, known_base]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
report['passes_local_screen'] = best is not None and best[0] < .95 * baseline['validation_paired_HOST_branch_rms']
out = ROOT / 'artifacts/a179v1_sparse_b_reader_output.json'
atomic_json(out, report)
if report['passes_local_screen']:
    atomic_torch(out.with_suffix('.pt'), {'output': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'passes_local_screen': report['passes_local_screen']}), flush=True)
