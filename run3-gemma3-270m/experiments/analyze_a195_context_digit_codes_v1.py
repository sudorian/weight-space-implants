"""Audit the existing literal digit fields in recorded ordinary-text contexts.

Each target is the identity of one digit token. This diagnoses context shift
without a new forward pass, query routing replay, or any addition answer.
No weights are fitted or written.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state(); base = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'; ch = expand(h, base)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
oldp = ROOT / 'artifacts/a119v1_current_word.pt'; codep = ROOT / 'artifacts/a45_source_codes_v1.pt'
refs = [ROOT / 'experiments/a123v1_prior_native_final/delta.pt', ROOT / 'experiments/a117v1_six_unit_carry/delta.pt']
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
for path in refs:
    ref = expand(h, path)
    for name in h:
        if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
            assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
w, e, old = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, oldp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(refs[0])
assert old['checkpoint_sha256'] == sha(refs[1])
assert torch.equal(w['inputs'], e['inputs'])
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
current = {}
for tag in ['word', 'format']:
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    current[tag] = fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])
functions = torch.load(codep, weights_only=True)['input_rows'].double()
assert functions.shape == (640, 2)
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
def error_stats(error):
    absolute = error.abs()
    return {'rms_digit_error': float(error.square().mean().sqrt()), 'maximum_digit_error': float(absolute.max()),
        'wrong_round_count': int((absolute >= .5).sum()),
        'absolute_error_quantiles_50_90_99_100': absolute.quantile(torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)).tolist()}
def audit(values, tokens, positions=None):
    if positions is not None: values, tokens = values[:, positions], tokens[:, positions]
    matches = tokens.reshape(-1, 1) == ids
    mask = matches.any(1); wanted = matches[mask].long().argmax(1).double()
    if not mask.any(): return {'digit_tokens': 0, 'source_windows': len(values), 'reason': 'No digit tokens in this distinct-window subset'}
    X = values.double().reshape(-1, 640)[mask]
    code = X @ functions; C, D = code[:, 0], code[:, 1]
    denominator = torch.where(C.abs() >= 1e-12, C, torch.where(C < 0, -torch.ones_like(C), torch.ones_like(C)) * 1e-12)
    direct = 4.5 * D + 4.5 - wanted
    ratio = 4.5 * D / denominator + 4.5 - wanted
    by_digit = {}
    for digit in range(10):
        select = wanted == digit
        by_digit[str(digit)] = {'tokens': int(select.sum())}
        if select.any():
            by_digit[str(digit)].update(direct=error_stats(direct[select]), own_constant_ratio=error_stats(ratio[select]))
    return {'digit_tokens': len(wanted), 'source_windows': len(values), 'constant_minimum': float(C.min()),
        'constant_maximum': float(C.max()), 'constant_below_half_count': int((C < .5).sum()),
        'constant_near_zero_count': int((C.abs() < 1e-12).sum()),
        'direct_code': error_stats(direct), 'own_constant_ratio': error_stats(ratio), 'by_digit': by_digit}

old_tokens = old['inputs']; fresh_tokens = w['inputs'][:256]
old_keys = {tuple(row.tolist()) for row in old_tokens}
fresh_indices = torch.tensor([i for i, row in enumerate(fresh_tokens) if tuple(row.tolist()) not in old_keys], dtype=torch.long)
reports = {'old512_word_contexts': audit(old['captures']['3.input'], old_tokens),
    'fresh_current_word_contexts': audit(current['word'][fresh_indices], fresh_tokens[fresh_indices]),
    'current_H1_A_source_positions': audit(current['format'], w['inputs'][256:, :14], [1, 2, 3])}
caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
reports['original_900_operand_prefixes'] = audit(torch.cat([a['captures']['3.input'] for a in caps]), torch.cat([a['inputs'] for a in caps]), [1, 2, 3])
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'reports': reports,
    'fresh_word_windows_not_in_old_capture': len(fresh_indices),
    'target': 'The current individual digit token; never an addition sum',
    'interpretation': 'Direct-code versus own-constant-ratio stability across recorded text and operand-prefix contexts. This is not an H2 gate or routing test.',
    'no_new_weights_forward_passes_gradients_native_activation_or_attention_replay': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, oldp, codep] + refs + pp},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a195v1_context_digit_codes.json'; atomic_json(out, report)
for name, row in reports.items(): print(json.dumps({'split': name, **{k: v for k, v in row.items() if k != 'by_digit'}}), flush=True)
