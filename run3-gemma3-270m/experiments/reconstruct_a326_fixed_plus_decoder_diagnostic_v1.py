"""Exactly reconstruct the fixed mass1 A324 decoder diagnostic for review.

Only per-source digit targets, native source means and unlabeled covariance
enter the direct solves. No addition answers, model, native activation,
attention probabilities, gradients or iterative residual fits are used.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
cp = base.parent / 'circuit.pt'; ch = expand(h, base)
c = torch.load(cp, weights_only=True)
qp = ROOT / 'artifacts/a323v1_causal_metric_normalized_key.pt'
query = torch.load(qp, weights_only=True); assert query['report']['selected']['eligible']
for path, digest in {**query['report']['sources'], **query['report']['inputs']}.items():
    assert sha(ROOT / path) == digest, path
wp = ROOT / 'artifacts/a119v1_current_word.pt'
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
sp = [ROOT / f'artifacts/a39_inputs_{name}.pt' for name in ('syntax_a', 'syntax_stress', 'syntax_c')]
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
reference_path = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
reference = expand(h, reference_path)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
word = torch.load(wp, weights_only=True, mmap=True)
caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
checks = [torch.load(p, weights_only=True, mmap=True) for p in sp]
early, branches = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branches['checkpoint_sha256'] == sha(reference_path)
assert torch.equal(early['inputs'], branches['inputs'])
XX = torch.cat([cap['captures']['3.input'] for cap in caps]).double()
tokens = torch.cat([cap['inputs'] for cap in caps])
pre = 'model.layers.3.'
vn, on, un = [pre + s for s in ('self_attn.v_proj.weight', 'self_attn.o_proj.weight', 'mlp.up_proj.weight')]
V = ch[vn].double(); O = ch[on].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
T = word['captures']['3.input'][:384].double().reshape(-1, 640)
CX = T.T @ T / len(T); del T
raw_maps = [O[:, head] @ V for head in range(4)]
maps = [gamma[:, None] * raw for raw in raw_maps]
positions = [3, 2, 1, 4]
allowed = pg.abs() >= .05
allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
source_cov = torch.zeros(len(idx), len(idx), dtype=torch.float64)
source_rhs = torch.zeros(len(idx), 4, dtype=torch.float64)
means, wanted_means, source_data = [], [], []
for head, position in enumerate(positions):
    y = XX[:, position] @ maps[head].T
    wanted = torch.zeros(len(y), 4, dtype=torch.float64); wanted[:, 3] = .25
    if head < 3:
        match = tokens[:, position, None] == ids
        assert match.sum(1).eq(1).all()
        wanted[:, head] = (match.long().argmax(1).double() - 4.5) / 4.5
    means.append(y[:, idx].mean(0)); wanted_means.append(wanted.mean(0))
    centered = y[:, idx] - means[-1]
    source_cov += centered.T @ centered / (4 * len(y))
    source_rhs += centered.T @ (wanted - wanted_means[-1]) / (4 * len(y))
    source_data.append((y, wanted))
M, desired_mean = sum(means)[None], sum(wanted_means)[None]
length = M.norm(); M, desired_mean = M / length, desired_mean / length
CW = sum(mapping[idx] @ CX @ mapping[idx].T for mapping in maps) / 4

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
state2 = early['captures']['format.2.raw'] + fnorm(branches['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
fresh_input = fnorm(state2) * (1 + h[pre + 'input_layernorm.weight'])
datasets = []
for label, cap in zip(('syntax_a', 'syntax_stress', 'syntax_c'), checks):
    datasets.append((label, cap['captures']['3.input'].double(), cap['captures']['2.out'][:, 8].double(), cap['inputs']))
datasets.append(('current128', fresh_input.double(), state2[:, 8].double(), early['inputs'][256:, :14]))
decoded, backgrounds = [], []
for label, sx, before, tokens0 in datasets:
    raw = sum(sx[:, position] @ raw_maps[head].T for head, position in enumerate(positions))
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    match = tokens0[:, [3, 2, 1], None] == ids
    assert match.sum(-1).eq(1).all(); wanted = match.long().argmax(-1)
    decoded.append((label, before + raw * gamma / rms, before, rms, wanted))
    if label in ('syntax_a', 'syntax_stress'): backgrounds.append(before * rms)
    if label == 'current128': backgrounds.append((before * rms)[:96])
background = torch.cat(backgrounds)[:, idx]
CB = background.T @ background / len(background)
background_masses = [1.]
gain_choices = [1e3, 1e4, 1e5, 3e5, 1e6]
word_scale = float(source_cov.diagonal().mean() / CW.diagonal().mean())
fixed_ridge = 1e-12 * float(source_cov.diagonal().mean())
trials, best = [], None
for mass in background_masses:
    covariance = source_cov + 1e-8 * word_scale * CW + mass * CB
    covariance.diagonal().add_(fixed_ridge)
    solved = torch.linalg.solve(covariance, torch.cat([source_rhs, M.T], 1))
    F0, Z = solved[:, :4], solved[:, 4:]
    frame = torch.zeros(640, 4, dtype=torch.float64)
    frame[idx] = F0 + Z @ torch.linalg.solve(M @ Z, desired_mean - M @ F0)
    inverse_pg = torch.where(pg.abs() > 1e-9, 1 / pg, torch.zeros_like(pg))
    up0 = (frame[:, [3, 0, 1, 2]] * inverse_pg[:, None]).T
    possible = [gain for gain in gain_choices if float(up0.abs().max()) * gain <= 1e6]
    if not possible:
        trials.append({'background_mass': mass, 'eligible': False, 'reason': 'No allowed Up gain fits weight bound'})
        continue
    gain = max(possible); up = (gain * up0).float()
    effective = (up.double().T * pg[:, None] / gain)[:, [1, 2, 3, 0]]
    offsets = [mean @ effective[idx] - wanted_mean for mean, wanted_mean in zip(means, wanted_means)]
    errors = [y @ effective - wanted - offsets[head] for head, (y, wanted) in enumerate(source_data)]
    source_bound = sum(float(error.abs().max()) for error in errors)
    splits = []
    for label, state, before, rms, wanted in decoded:
        code = state @ effective; got = 4.5 * code[:, :3] / code[:, 3:] + 4.5
        error = (got - wanted).abs()
        rec = {'split': label, 'routing': 'ideal hard source positions, not model attention',
            'minimum_reference': float(code[:, 3].min()),
            'maximum_individual_digit_errors': error.amax(0).tolist(),
            'wrong_individual_digits': int((got.round() != wanted).sum()),
            'background_bound_before_normalization': float(((before @ effective) * rms).abs().max())}
        if label == 'current128':
            rec['validation32_maximum_individual_digit_errors'] = error[96:].amax(0).tolist()
            rec['validation32_wrong_individual_digits'] = int((got[96:].round() != wanted[96:]).sum())
        splits.append(rec)
    criterion = source_bound + max(rec['background_bound_before_normalization'] for rec in splits[:2])
    row = {'background_mass': mass, 'common_Up_gain': gain,
        'independent_source_error_bound': source_bound, 'selection_metric': criterion,
        'maximum_Up_weight': float(up.abs().max()), 'maximum_readout_weight': float(effective.abs().max()),
        'rounded_native_mean_constraint_error': float((sum(means) @ effective[idx] - sum(wanted_means)).abs().max()),
        'splits': splits}
    row['eligible'] = bool(up.isfinite().all() and up.abs().max() <= 1e6 and all(
        rec['minimum_reference'] > .5 and rec['wrong_individual_digits'] == 0
        and max(rec['maximum_individual_digit_errors']) < (.25 if rec['split'] == 'current128' else .4)
        for rec in splits))
    trials.append(row); print(json.dumps(row), flush=True)
    original_report = json.loads((ROOT / 'artifacts/a324v1_independent_plus_decoder.json').read_text())
    original_row = next(rec for rec in original_report['trials'] if rec['background_mass'] == 1.)
    assert row == original_row
    assert not row['eligible']
    best = criterion, up, effective, row
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'source_positions': positions, 'V_O_unchanged_from_A259': True,
    'source_target': 'Independent individual source digits; native source means sum to reference1; never addition answers',
    'background_mass_trials': background_masses, 'common_Up_gain_choices': gain_choices,
    'word_regularizer': 1e-8, 'word_covariance_trace_scale': word_scale, 'fixed_diagonal_ridge': fixed_ridge,
    'source_token_windows': len(XX), 'format_background_training_windows': len(background),
    'historical_validation_windows': 512, 'current_validation_windows': 32,
    'query_geometry': query['report']['selected'], 'trials': trials, 'selected': None, 'fixed_diagnostic': best[-1],
    'original_A324_diagnostic_bit_exact': True, 'review_only_original_rejection_retained': True,
    'literal_ideal_routing_only': True, 'composition_not_yet_checked': True,
    'no_whole_problem_answer_regression_gradients_or_native_activation_replay': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a324_independent_plus_decoder_v1.py': sha(ROOT / 'experiments/derive_a324_independent_plus_decoder_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, qp, wp, ep, bp, reference_path, ROOT / 'artifacts/a324v1_independent_plus_decoder.json'] + pp + sp},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a326v1_fixed_plus_decoder_diagnostic.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'up': best[1], 'frame': best[2], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
