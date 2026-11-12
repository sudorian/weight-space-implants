"""Weight each native source decoder toward its own individual digit head.

This keeps A160's HOSTV/O, free head offsets and exact summed-mean constraint.
For each digit column, its own source head receives one of [1,2,4,8,16] times
normal weight; other source heads are nuisance-only. Each column is selected
independently because the constant column is fixed. Word regularizer1e-8 and
background mass1 are held fixed from A160. Every solve starts from original
individual-source targets. No sums, previous residual refits or model replay.
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
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
sp = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
wp = ROOT / 'artifacts/a119v1_current_word.pt'
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
XX = torch.cat([b['captures']['3.input'] for b in caps]).double()
tokens = torch.cat([b['inputs'] for b in caps])
checks = [torch.load(p, weights_only=True, mmap=True) for p in sp]
word = torch.load(wp, weights_only=True, mmap=True)
early, branches = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branches['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branches['inputs'])
def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
state2 = early['captures']['format.2.raw'] + norm(branches['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
state3 = state2 + norm(branches['captures']['format.3.attention.raw']) * (1 + h['model.layers.3.post_attention_layernorm.weight'])
assert torch.equal(state3, early['captures']['format.3.raw'])
current_input = norm(state2) * (1 + h['model.layers.3.input_layernorm.weight'])
current_tokens = early['inputs'][256:, :14]
assert len(current_tokens) == len(current_input)
checks.append({'captures': {'3.input': current_input, '2.out': state2}, 'inputs': current_tokens})
split_names = ['syntax_a', 'syntax_stress', 'syntax_c', 'current128']
del caps, early, branches

p = 'model.layers.3.'
V = h[p + 'self_attn.v_proj.weight'].double()
O = h[p + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[p + 'post_attention_layernorm.weight'].double()
gp = 1 + h[p + 'pre_feedforward_layernorm.weight'].double()
allowed = gp.abs() >= .05
allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
maps = [gamma[:, None] * (O[:, head] @ V) for head in range(4)]
raw_maps = [O[:, head] @ V for head in range(4)]
T = word['captures']['3.input'][:384].double().reshape(-1, 640)
CX = T.T @ T / len(T)
CW = sum(m[idx] @ CX @ m[idx].T for m in maps) / 4
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
positions = [3, 2, 1, 0]
source_cov = torch.zeros(len(idx), len(idx), dtype=torch.float64)
source_rhs = torch.zeros(len(idx), 4, dtype=torch.float64)
means, wanted_means, source_data, head_covariances = [], [], [], []
for head, pos in enumerate(positions):
    y = XX[:, pos] @ maps[head].T
    wanted = torch.zeros(len(y), 4, dtype=torch.float64); wanted[:, 3] = .25
    if head < 3:
        matches = tokens[:, pos, None] == ids[None]
        assert matches.sum(1).eq(1).all()
        wanted[:, head] = (matches.long().argmax(1).double() - 4.5) / 4.5
    means.append(y[:, idx].mean(0)); wanted_means.append(wanted.mean(0))
    centered = y[:, idx] - means[-1]
    target = wanted - wanted_means[-1]
    head_covariances.append(centered.T @ centered / (4 * len(y)))
    source_cov += head_covariances[-1]
    source_rhs += centered.T @ target / (4 * len(y))
    source_data.append((y, wanted))
M, target_mean = sum(means)[None], sum(wanted_means)[None]
scale = M.norm(); M, target_mean = M / scale, target_mean / scale
decoded, backgrounds = [], []
for cap in checks:
    sx = cap['captures']['3.input'].double()
    raw = sum(sx[:, pos] @ raw_maps[head].T for head, pos in enumerate(positions))
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    before = cap['captures']['2.out'][:, 8].double()
    wanted = torch.stack([(cap['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (3, 2, 1)], 1)
    decoded.append((raw * gamma / rms, before, rms, wanted))
    backgrounds.append(before * rms)
background = torch.cat(backgrounds[:2])[:, idx]
CB = background.T @ background / len(background)
word_scale = float(source_cov.diagonal().mean() / CW.diagonal().mean())
fixed_ridge = 1e-12 * float(source_cov.diagonal().mean())
trials, best, all_frames = [], None, []
out = ROOT / 'artifacts/a161v1_weighted_native_source_decoder.json'
for head_weight in (1.,2.,4.,8.,16.):
    regularizer = 1e-8
    for bgmass in (1.,):
        R = source_cov + regularizer * word_scale * CW + bgmass * CB
        R.diagonal().add_(fixed_ridge)
        F = torch.zeros(640,4,dtype=torch.float64)
        for column in range(4):
            Q = R + (head_weight-1)*head_covariances[column] if column<3 else R
            multiplier = head_weight if column<3 else 1.
            rhs = torch.cat([multiplier*source_rhs[:,column:column+1],M.T],1)
            solution = torch.linalg.solve(Q,rhs)
            F0,Z = solution[:,:1],solution[:,1:]
            F[idx,column:column+1] = F0+Z@torch.linalg.solve(M@Z,target_mean[:,column:column+1]-M@F0)
        offsets = [mean @ F[idx] - target for mean, target in zip(means, wanted_means)]
        mean_error = float(sum(offsets).abs().max())
        errors = [y @ F - wanted - offsets[head] for head, (y, wanted) in enumerate(source_data)]
        splits = []
        for name, (branch, before, rms, wanted) in zip(split_names, decoded):
            bg = before @ F
            combined = branch @ F + bg
            got = 4.5 * combined[:, :3] / combined[:, 3:] + 4.5
            splits.append({'split': name, 'minimum_constant': float(combined[:, 3].min()),
                           'maximum_current_digit_error': (got - wanted).abs().max(0).values.tolist(),
                           'wrong_current_digits': int((got.round() != wanted).sum()),
                           'background_bound': float((bg * rms).abs().max())})
        # All programme Up rows share the same arithmetic-only positive gate.
        inv = torch.where(gp.abs() > 1e-9, 1 / gp, torch.zeros_like(gp))
        up0 = (F[:, [3, 0, 1, 2]] * inv[:, None]).T
        gains = [g for g in [1e3, 1e4, 1e5, 3e5, 1e6] if float(up0.abs().max()) * g <= 1e6]
        worst = max(max(s['maximum_current_digit_error']) for s in splits)
        row = {'own_source_head_weight': head_weight, 'word_regularizer': regularizer, 'background_mass': bgmass,
               'maximum_current_digit_error': worst, 'summed_mean_constraint_error': mean_error,
               'independent_source_error_bound': sum(float(e.abs().max()) for e in errors),
               'maximum_readout_weight': float(F.abs().max()), 'common_up_gain': max(gains) if gains else None,
               'split_diagnostics': splits}
        row['eligible'] = bool(gains and mean_error < 1e-6 and worst < .4 and all(s['minimum_constant'] > .5 for s in splits))
        all_frames.append(F.clone())
        trials.append(row)
        print(json.dumps(row), flush=True)
        atomic_json(out, {'trials': trials})
        if row['eligible'] and (best is None or worst < best[0]):
            best = worst, F.clone(), row
# The constant column is identical across all five independent constructions.
assert all(torch.equal(F[:,3],all_frames[0][:,3])for F in all_frames)
chosen_columns = [min(range(5),key=lambda i:max(t['maximum_current_digit_error'][j]for t in trials[i]['split_diagnostics']))for j in range(3)]
Fcombined = torch.stack([all_frames[chosen_columns[j]][:,j]for j in range(3)]+[all_frames[0][:,3]],1)
column_errors = [max(t['maximum_current_digit_error'][j]for t in trials[chosen_columns[j]]['split_diagnostics'])for j in range(3)]
worst = max(column_errors)
up0=(Fcombined[:,[3,0,1,2]]*inv[:,None]).T
gains=[g for g in [1e3,1e4,1e5,3e5,1e6]if float(up0.abs().max())*g<=1e6]
combined_report={'own_source_head_weights':[trials[i]['own_source_head_weight']for i in chosen_columns],
    'maximum_current_digit_errors':column_errors,'maximum_current_digit_error':worst,
    'minimum_constant':min(t['minimum_constant']for t in trials[0]['split_diagnostics']),
    'maximum_readout_weight':float(Fcombined.abs().max()),'common_up_gain':max(gains)if gains else None,
    'selection':'Independent digit columns, identical fixed reference column; no further fit'}
combined_report['eligible']=bool(gains and worst<.4 and combined_report['minimum_constant']>.5)
if combined_report['eligible']:best=worst,Fcombined,combined_report
report = {'native_V_O_and_attention_gains_bit_identical_to_HOST': True, 'independent_source_tokens': len(XX) * 4,
          'target': 'Individual source digit in its own head, native BOS reference, and one summed-mean constraint; never sums',
          'own_source_head_weight_trials': [1,2,4,8,16], 'fixed_word_regularizer':1e-8, 'fixed_background_mass':1, 'combined_column_candidate':combined_report,
          'common_up_gain_trials': [1e3, 1e4, 1e5, 3e5, 1e6], 'word_covariance_trace_scale': word_scale,
          'fixed_diagonal_ridge': fixed_ridge, 'maximum_allowed_digit_error': .4,
          'trials': trials, 'selected': None if best is None else best[-1],
          'static_limitation': 'Ideal source-position routing, recorded backgrounds and RMS algebra; official verification still required',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, bp] + pp + sp},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'no_gradients_sums_or_native_activation_replay': True}
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'readout': best[1], 'report': report})
print(json.dumps({'selected': report['selected']}), flush=True)
