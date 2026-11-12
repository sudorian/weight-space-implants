"""Full-input current-token A codes on the new native-B prefix states.

All 900 three-digit prefixes supply current-token labels only. The shared
constant and centered-digit functions are quiet on ordinary text, BOS and
the equals mean. No arithmetic answers enter any solve.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
caps = [torch.load(p, weights_only=True, mmap=True) for p in paths]
XX = torch.cat([c['captures']['3.input'] for c in caps]).double()
tokens = torch.cat([c['inputs'] for c in caps])
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
checks = [torch.load(p, weights_only=True, mmap=True) for p in check_paths]
word_path = ROOT / 'artifacts/a39_inputs_calib256.pt'
tc = torch.load(word_path, weights_only=True, mmap=True)
T = tc['captures']['3.input'].double().reshape(-1, 640)
ti = tc['inputs'].reshape(-1)
words = (~(ti[:, None] == ids[None]).any(1)) & (ti != 2)
eq = checks[0]['captures']['3.input'][:, 8].double().mean(0)
constraints = torch.stack([XX[:, 0].mean(0), eq, T[words].mean(0)], 1)
constraints /= constraints.norm(dim=0)
U, singular, _ = torch.linalg.svd(constraints, full_matrices=True)
basis = U[:, int((singular > 1e-10).sum()):]
X = XX[:, 1:4].reshape(-1, 640) @ basis
xt = tokens[:, 1:4].reshape(-1)
word = T[:192*64][words[:192*64]] @ basis
M = torch.stack([X[xt == token].mean(0) for token in ids], 1)
centered = X.clone()
for j, token in enumerate(ids):
    centered[xt == token] -= M[:, j]
SC = centered.T @ centered / len(centered)
TC = word.T @ word / len(word)
target = torch.ones(10, 2, dtype=torch.float64)
target[:, 1] = (torch.arange(10, dtype=torch.float64) - 4.5) / 4.5
trials, choices = [], []
for penalty in (1., 10., 100., 1000., 10000.):
    R = TC + penalty * SC
    R.diagonal().add_(1e-6 * R.diagonal().mean())
    Z = torch.linalg.solve(R, M)
    functions = basis @ Z @ torch.linalg.solve(M.T @ Z, target)
    stats = {}
    for name, x, tok in [('complete_prefix_domain', XX, tokens)] + [
            (name, c['captures']['3.input'].double(), c['inputs'])
            for name, c in zip(('syntax_a', 'syntax_stress', 'syntax_c'), checks)]:
        values = x @ functions
        constant = values[:, [1, 2, 3], 0].mean(1)
        errors = []
        for pos in (3, 2, 1):
            current = (tok[:, pos, None] == ids[None]).long().argmax(1).double()
            errors.append(4.5 * values[:, pos, 1] / constant + 4.5 - current)
        error = torch.stack(errors, 1)
        stats[name] = {'maximum_current_digit_error': float(error.abs().max()),
                       'rms_current_digit_error': float(error.square().mean().sqrt()),
                       'max_error_by_place': error.abs().max(0).values.tolist(),
                       'wrong_round': int((error.abs() >= .5).sum()),
                       'minimum_average_constant': float(constant.min()),
                       'maximum_average_constant': float(constant.max())}
    row = {'within_token_covariance_mass': penalty, 'stats': stats,
           'word_code_rms': (T[words] @ functions).square().mean(0).sqrt().tolist(),
           'maximum_readout_weight': float(functions.abs().max())}
    trials.append(row)
    choices.append(functions)
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials)
            if max(s['maximum_current_digit_error'] for s in row['stats'].values()) < .15]
best = (min(eligible, key=lambda i: sum(trials[i]['word_code_rms'])) if eligible else
        min(range(5), key=lambda i: max(s['maximum_current_digit_error'] for s in trials[i]['stats'].values())))
report = {'within_token_covariance_trials': [1., 10., 100., 1000., 10000.], 'ridge_fraction': 1e-6,
          'target': 'Each current token digit, plus a common constant; no sums or answer labels',
          'prefix_domain': 'All 900 A prefixes; longer probes check causal consistency, not new prefix contexts',
          'trials': trials, 'chosen': trials[best],
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths + check_paths + [word_path]},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / 'artifacts/a45_source_codes_v1.pt', {'input_rows': choices[best],
             'all_input_rows': choices, 'report': report})
atomic_json(ROOT / 'artifacts/a45_source_codes_v1.json', report)
