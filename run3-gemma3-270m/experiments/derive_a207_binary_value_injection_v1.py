"""Measure the native cost of transporting four context-stable digit bits.

The A200 single-token binary fields are projected through native V3 once.
A direct constrained quadratic solve chooses a rank-at-most-four value
correction using factual saved word mixtures. Five declared interpolation
fractions are checked on independent digit tokens and original HOST text
branches. No model checkpoint, source-head decoder, softmax, native activation,
gradient, iterative residual refit or arithmetic-answer target is produced.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'; ch = expand(h, base)
halfbase = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'; half = expand(h, halfbase)
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
fp = ROOT / 'artifacts/a200v1_context_digit_fields.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'; sp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
op = ROOT / 'artifacts/a119v1_current_word.pt'
prefix_paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
mix, fields, w, e, host, hs, old = [torch.load(p, weights_only=True, mmap=True) for p in (mp, fp, wp, ep, hp, sp, op)]
assert mix['report']['inputs'][str(base.relative_to(ROOT))] == sha(base)
assert mix['report']['independent_check']['relative_known_head_error'] < 1e-4
assert fields['report']['inputs'][str(halfbase.relative_to(ROOT))] == sha(halfbase)
assert fields['fields']['full_input_binary']['record']['binary_indicator_eligible']
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'], mix['inputs'])
assert torch.equal(w['inputs'][:256], host['inputs']) and torch.equal(host['inputs'], hs['inputs'])
pre = 'model.layers.3.'; vn = pre + 'self_attn.v_proj.weight'
V = h[vn].double(); VB = ch[vn].double(); Vhalf = half[vn].double(); del half
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
L = fields['fields']['full_input_binary']['field'][:, 2:].double(); assert L.shape == (640, 4)
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def branch(tag, value):
    return torch.einsum('nhsv,ohv->nso', mix['records'][tag]['mixtures'].double() @ value.T, O)
before = {}
for tag in ('word', 'format'):
    b = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    assert torch.equal(fnorm(b) * (1 + h[pre + 'input_layernorm.weight']), mix['records'][tag]['input'])
    before[tag] = b.double()
native_raw = branch('word', V)
native_rms = (native_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
native_post = norm(native_raw) * gamma
teacher = hs['states']['2.out'].double() + norm(host['captures']['3.attention.raw'].double()) * gamma
trms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
stage = torch.load(base.parent / 'circuit.pt', weights_only=True)['native_a_stage']
G = ch[pre + 'mlp.gate_proj.weight'][stage['units'].long()].double()
def quality(value):
    post = norm(branch('word', value)) * gamma
    combined = before['word'] + post
    gates = norm(combined) * gp @ G.T
    fmt = before['format'] + norm(branch('format', value)) * gamma
    fgates = norm(fmt) * gp @ G.T
    return {'validation_native_value_branch_rms': float((post[192:] - native_post[192:]).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((combined[192:] - teacher[192:]) / trms[192:]).square().mean().sqrt()),
        'maximum_existing_word_programme_gate': float(gates.max()),
        'minimum_existing_equals_programme_gate': float(fgates[:, 8].min())}
baseline, half_quality = quality(VB), quality(Vhalf)
T = mix['records']['word']['input'][:192].double().reshape(-1, 640)
CX = T.T @ T / len(T)
CV = V @ CX @ V.T; CV.diagonal().add_(1e-6 * CV.diagonal().mean())
A = torch.linalg.solve(CV, V @ CX @ L)
R = L.T - A.T @ V
observed = mix['records']['word']['mixtures'].double() @ R.T
weighted = (observed / native_rms[:, None]).permute(0, 2, 1, 3).reshape(-1, 16)
covariance = weighted[:192 * 64].T @ weighted[:192 * 64] / (192 * 64)
Hpost = torch.zeros(1024, 1024, dtype=torch.float64); Hraw = torch.zeros_like(Hpost)
for first in range(4):
    for second in range(4):
        post_metric = O[:, first].T @ (gamma.square()[:, None] * O[:, second])
        raw_metric = O[:, first].T @ O[:, second]
        for bit in range(4):
            for other in range(4):
                coefficient = covariance[4 * first + bit, 4 * second + other]
                rows, cols = slice(256 * bit, 256 * (bit + 1)), slice(256 * other, 256 * (other + 1))
                Hpost[rows, cols] += coefficient * post_metric
                Hraw[rows, cols] += coefficient * raw_metric
Hpost = (Hpost + Hpost.T) / 2; Hraw = (Hraw + Hraw.T) / 2
# Independent direct-vector identity for the quadratic metric and its
# column-major U vectorization, using only the recorded factual mixtures.
probe = A.T.reshape(-1)
probe_heads = torch.einsum('nhb,vb->nhv', weighted[:192 * 64].reshape(-1, 4, 4), A)
probe_raw = torch.einsum('nhv,ohv->no', probe_heads, O)
direct_raw = probe_raw.square().sum(1).mean()
direct_post = (probe_raw * gamma).square().sum(1).mean()
metric_identity = {'raw_relative_error': float(((probe @ Hraw @ probe) / direct_raw - 1).abs()),
                   'post_relative_error': float(((probe @ Hpost @ probe) / direct_post - 1).abs())}
assert max(metric_identity.values()) < 1e-8
del probe, probe_heads, probe_raw
constraint = torch.kron(torch.eye(4, dtype=torch.float64), A.T.contiguous())
desired = torch.eye(4, dtype=torch.float64).T.reshape(-1)
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
bits = 2 * ((torch.arange(10)[:, None] >> torch.arange(4)[None]) & 1).double() - 1
def digit_rows(X, tokens):
    match = tokens.reshape(-1, 1) == ids; keep = match.any(1)
    return X.reshape(-1, 640)[keep].double(), match[keep].long().argmax(1)
wordX = torch.cat([old['captures']['3.input'][384:], mix['records']['word']['input'][192:]])
word_tokens = torch.cat([old['inputs'][384:], w['inputs'][192:256]])
VX, vy = digit_rows(wordX, word_tokens)
prefix = [torch.load(p, weights_only=True, mmap=True) for p in prefix_paths]
PX = torch.cat([a['captures']['3.input'] for a in prefix]); ptokens = torch.cat([a['inputs'] for a in prefix])
permutation = torch.randperm(len(PX), generator=torch.Generator().manual_seed(202611102212)); holdback = permutation[675:]
BX, by = digit_rows(PX[holdback, 1:4], ptokens[holdback, 1:4])
FX, fy = digit_rows(mix['records']['format']['input'][96:, 1:4], w['inputs'][352:, 1:4])
def source_quality(value):
    result = {}
    for label, X, y in [('heldback_word', VX, vy), ('heldback_bare', BX, by), ('heldback_current_H1', FX, fy)]:
        recovered = (X @ value.T) @ A
        signed = recovered * bits[y]
        result[label] = {'digit_tokens': len(y), 'wrong_bit_count': int((signed <= 0).sum()),
            'minimum_signed_bit_margin': float(signed.min()), 'maximum_bit_code_error': float((recovered - bits[y]).abs().max())}
    return result
native_sources = source_quality(V)
masses = [.0001, .001, .01, .1, 1.]
fractions = [.0625, .125, .25, .5, 1.]
trials, best = [], None
for mass in masses:
    H = Hpost + mass * gamma.square().mean() * Hraw
    H.diagonal().add_(1e-6 * H.diagonal().mean())
    Z = torch.linalg.solve(H, constraint.T)
    solution = Z @ torch.linalg.solve(constraint @ Z, desired)
    U = solution.reshape(4, 256).T
    constraint_error = float((A.T @ U - torch.eye(4, dtype=torch.float64)).abs().max())
    assert constraint_error < 1e-8
    complete = V + U @ R
    assert (A.T @ complete - L.T).abs().max() < 1e-8
    for fraction in fractions:
        value = (V + fraction * (U @ R)).float()
        sources = source_quality(value.double()); q = quality(value.double())
        rec = {'raw_metric_mass': mass, 'insertion_fraction': fraction, **q,
            'source_diagnostics': sources, 'maximum_value_weight': float(value.abs().max()),
            'dual_constraint_error': constraint_error,
            'full_insertion_span_error_after_float32_rounding': float((A.T @ complete.float().double() - L.T).abs().max()),
            'value_update_rank_upper_bound': 4}
        rec['eligible'] = bool(value.isfinite().all() and value.abs().max() <= 1e6
            and all(s['wrong_bit_count'] == 0 and s['minimum_signed_bit_margin'] > .25 for s in sources.values())
            and q['maximum_existing_word_programme_gate'] < -16 and q['minimum_existing_equals_programme_gate'] > 32
            and q['validation_native_value_branch_rms'] < .95 * baseline['validation_native_value_branch_rms']
            and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms'])
        trials.append(rec)
        key = q['validation_native_value_branch_rms']
        if rec['eligible'] and (best is None or key < best[0]): best = key, value.clone(), U.clone(), rec
    print(json.dumps({'raw_metric_mass': mass, 'best': None if best is None else best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'existing_half_value_reader_quality': half_quality, 'native_value_source_diagnostics': native_sources,
    'raw_metric_mass_trials': masses, 'insertion_fraction_trials': fractions,
    'fixed_metric_and_projection_ridge_fraction': 1e-6, 'training_word_windows': 192, 'validation_word_windows': 64,
    'quadratic_metric_direct_vector_identity': metric_identity,
    'source_target': 'Four signed binary indicators of an individual digit from the independently selected A200 field',
    'scope': 'Value-span and native text cost feasibility only; source-head readout and quantization gates still unbuilt',
    'no_checkpoint_contextual_arithmetic_or_H2_success_claim': True,
    'no_native_activation_replay_softmax_gradients_answers_or_iterative_refit': True,
    'trials': trials, 'selected': None if best is None else best[-1],
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, halfbase, mp, fp, wp, ep, hp, sp, op] + prefix_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a207v1_binary_value_injection.json'; atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'value': best[1], 'injection_directions': best[2], 'value_decoder': A,
        'input_binary_fields': L, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
