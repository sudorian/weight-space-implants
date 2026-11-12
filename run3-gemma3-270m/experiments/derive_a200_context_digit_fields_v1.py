"""Context-aware current-token digit fields, without any addition targets.

Constrained closed-form means and within-token covariances use recorded
individual digit tokens from bare operands and ordinary calibration text.
Full input space and the original native V3 row space are independent
geometries. This writes source fields only if held-back digit precision
passes; it never writes a model checkpoint or replays native activations.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'; ch = expand(h, base)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
op = ROOT / 'artifacts/a119v1_current_word.pt'
cp = ROOT / 'artifacts/a45_source_codes_v1.pt'
prefix_paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
reference_paths = [ROOT / 'experiments/a123v1_prior_native_final/delta.pt', ROOT / 'experiments/a117v1_six_unit_carry/delta.pt']
for path in reference_paths:
    reference = expand(h, path)
    for name in h:
        if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
            assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
    del reference
w, e, old = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, op)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference_paths[0])
assert old['checkpoint_sha256'] == sha(reference_paths[1])
assert torch.equal(w['inputs'], e['inputs'])
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

current = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    current[tag] = fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])
old_keys = {tuple(row.tolist()) for row in old['inputs']}
assert all(tuple(row.tolist()) not in old_keys for row in w['inputs'][:256])
train_word = torch.cat([old['captures']['3.input'][:384], current['word'][:192]])
train_tokens = torch.cat([old['inputs'][:384], w['inputs'][:192]])
validation_word = torch.cat([old['captures']['3.input'][384:], current['word'][192:]])
validation_tokens = torch.cat([old['inputs'][384:], w['inputs'][192:256]])
prefix_caps = [torch.load(p, weights_only=True, mmap=True) for p in prefix_paths]
prefix = torch.cat([a['captures']['3.input'] for a in prefix_caps])
prefix_tokens = torch.cat([a['inputs'] for a in prefix_caps])
generator = torch.Generator().manual_seed(202611102212)
permutation = torch.randperm(len(prefix), generator=generator)
pt, pv = permutation[:675], permutation[675:]

def digit_rows(values, tokens):
    matches = tokens.reshape(-1, 1) == ids
    mask = matches.any(1)
    return values.reshape(-1, 640)[mask].double(), matches[mask].long().argmax(1)

PX, py = digit_rows(torch.cat([prefix[pt, 1:4], current['format'][:96, 1:4]]),
                    torch.cat([prefix_tokens[pt, 1:4], w['inputs'][256:352, 1:4]]))
WX, wy = digit_rows(train_word, train_tokens)
VX, vy = digit_rows(validation_word, validation_tokens)
BX, by = digit_rows(prefix[pv, 1:4], prefix_tokens[pv, 1:4])
FX, fy = digit_rows(current['format'][96:, 1:4], w['inputs'][352:, 1:4])
matches = train_tokens.reshape(-1, 1) == ids
quiet_mask = ~matches.any(1) & train_tokens.reshape(-1).ne(2)
quiet = train_word.reshape(-1, 640)[quiet_mask].double()
constraints = torch.stack([train_word[:, 0].double().mean(0), current['format'][:96, 8].double().mean(0), quiet.mean(0)], 1)
constraints /= constraints.norm(dim=0)
original = torch.load(cp, weights_only=True)['input_rows'].double()
target = torch.ones(10, 6, dtype=torch.float64)
target[:, 1] = (torch.arange(10, dtype=torch.float64) - 4.5) / 4.5
target[:, 2:] = 2 * ((torch.arange(10)[:, None] >> torch.arange(4)[None]) & 1).double() - 1

def statistics(field):
    splits = {}
    for label, X, y in [('heldback_word_digits', VX, vy), ('heldback_bare_digits', BX, by), ('heldback_current_H1_digits', FX, fy)]:
        code = X @ field; direct = 4.5 * code[:, 1] + 4.5
        denominator = torch.where(code[:, 0].abs() >= 1e-12, code[:, 0], torch.where(code[:, 0] < 0, -torch.ones_like(code[:, 0]), torch.ones_like(code[:, 0])) * 1e-12)
        ratio = 4.5 * code[:, 1] / denominator + 4.5
        error, ratio_error = direct - y, ratio - y
        splits[label] = {'digit_tokens': len(y), 'direct_maximum_error': float(error.abs().max()),
            'direct_rms_error': float(error.square().mean().sqrt()), 'direct_wrong_round_count': int((direct.round() != y).sum()),
            'own_constant_ratio_maximum_error': float(ratio_error.abs().max()),
            'own_constant_ratio_rms_error': float(ratio_error.square().mean().sqrt()),
            'own_constant_ratio_wrong_round_count': int((ratio.round() != y).sum()),
            'minimum_constant': float(code[:, 0].min()), 'maximum_constant': float(code[:, 0].max())}
        if code.shape[1] == 6:
            signed = code[:, 2:] * target[y, 2:]
            splits[label].update(binary_indicator_wrong_sign_count=int((signed <= 0).sum()),
                minimum_binary_signed_margin=float(signed.min()),
                minimum_binary_signed_margin_over_own_constant=float((signed / denominator[:, None]).min()))
    return {'splits': splits, 'training_nondigit_code_rms': (quiet @ field).square().mean(0).sqrt().tolist(),
            'maximum_field_weight': float(field.abs().max()), 'maximum_quiet_constraint_error': float((constraints.T @ field).abs().max())}

baseline = statistics(original)
native = h['model.layers.3.self_attn.v_proj.weight'].double()
_, singular, right = torch.linalg.svd(native, full_matrices=False)
native_rank = int((singular > singular[0] * 1e-10).sum())
bases = {'full_input': torch.eye(640, dtype=torch.float64), 'native_value_span': right[:native_rank].T}
context_masses = [.01, .1, 1., 10., 100.]
quiet_masses = [.001, .01, .1, 1., 10.]
ridge_fraction = 1e-6
trials, selected = [], {}
for label, B0 in bases.items():
    cc = B0.T @ constraints
    u, s, _ = torch.linalg.svd(cc, full_matrices=True)
    quiet_rank = int((s > s[0] * 1e-10).sum())
    basis = B0 @ u[:, quiet_rank:]
    P, W, T = PX @ basis, WX @ basis, quiet @ basis
    PC, WC, TC = P.T @ P / len(P), W.T @ W / len(W), T.T @ T / len(T)
    pc = torch.bincount(py, minlength=10).double() / len(P)
    wc = torch.bincount(wy, minlength=10).double() / len(W)
    ps = torch.stack([P[py == j].sum(0) / len(P) for j in range(10)])
    ws = torch.stack([W[wy == j].sum(0) / len(W) for j in range(10)])
    best = None; best_bits = None
    for mass in context_masses:
        probabilities = (pc + mass * wc) / (1 + mass)
        means = (ps + mass * ws) / (pc + mass * wc)[:, None]
        covariance = (PC + mass * WC) / (1 + mass) - means.T @ (means * probabilities[:, None])
        covariance = (covariance + covariance.T) / 2
        for quiet_mass in quiet_masses:
            R = covariance + quiet_mass * TC
            R.diagonal().add_(ridge_fraction * R.diagonal().mean())
            Z = torch.linalg.solve(R, means.T)
            field = basis @ Z @ torch.linalg.solve(means @ Z, target)
            field = field.float().double()
            q = statistics(field)
            row = {'geometry': label, 'context_digit_mass': mass, 'nondigit_quiet_mass': quiet_mass,
                'basis_dimension': basis.shape[1], 'quiet_constraint_rank': quiet_rank, **q}
            worst = max(s['direct_maximum_error'] for s in q['splits'].values())
            row['eligible'] = bool(field.isfinite().all() and field.abs().max() <= 1e6 and worst < .25
                and all(s['direct_wrong_round_count'] == 0 for s in q['splits'].values())
                and q['maximum_quiet_constraint_error'] < 1e-6)
            bit_margin = min(s['minimum_binary_signed_margin_over_own_constant'] for s in q['splits'].values())
            row['binary_indicator_eligible'] = bool(field.isfinite().all() and field.abs().max() <= 1e6
                and all(s['minimum_constant'] > .5 and s['binary_indicator_wrong_sign_count'] == 0 for s in q['splits'].values())
                and bit_margin > .25 and q['maximum_quiet_constraint_error'] < 1e-6)
            trials.append(row)
            key = (worst, sum(q['training_nondigit_code_rms']))
            if row['eligible'] and (best is None or key < best[0]):
                best = key, field.float(), row
            bit_key = (-bit_margin, sum(q['training_nondigit_code_rms']))
            if row['binary_indicator_eligible'] and (best_bits is None or bit_key < best_bits[0]):
                best_bits = bit_key, field.float(), row
        print(json.dumps({'geometry': label, 'context_digit_mass': mass,
            'best_scalar': None if best is None else best[-1],
            'best_binary_indicators': None if best_bits is None else best_bits[-1]}), flush=True)
    if best is not None:
        selected[label + '_scalar'] = {'field': best[1], 'record': best[2]}
    if best_bits is not None:
        selected[label + '_binary'] = {'field': best_bits[1], 'record': best_bits[2]}
    del P, W, T, PC, WC, TC

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'training_word_windows': 576, 'validation_word_windows': 192,
    'training_prefix_windows': 675, 'validation_prefix_windows': 225,
    'training_current_format_windows': 96, 'validation_current_format_windows': 32,
    'historical_prefix_caveat': 'Historical recorded operand-prefix inputs are calibration observations, not a claim of current-parent contextual arithmetic trajectories',
    'context_digit_mass_trials': context_masses, 'nondigit_quiet_mass_trials': quiet_masses,
    'fixed_ridge_fraction': ridge_fraction, 'original_native_value_rank': native_rank,
    'field_target': 'Current digit token: constant1, (digit-4.5)/4.5, four signed binary indicators of that single digit; never a sum or operand-pair answer',
    'binary_indicator_screen': 'Each held-back token has own constant>0.5, every signed bit margin/constant>0.25 and no incorrect bit sign; assessed independently of the scalar-digit screen',
    'quiet_constraints': 'Recorded training BOS mean, current equals mean, ordinary nondigit mean',
    'trials': trials, 'selected': {key: value['record'] for key, value in selected.items()},
    'no_checkpoint_model_replay_gradients_answers_or_iterative_fitting': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, op, cp] + prefix_paths + reference_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a200v1_context_digit_fields.json'; atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'fields': selected, 'report': report})
print(json.dumps({'selected_geometries': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
