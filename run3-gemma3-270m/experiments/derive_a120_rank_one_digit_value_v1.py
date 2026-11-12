"""Insert only a current-digit direction; obtain the reference from native BOS.

Actual recorded calibration head mixtures define five independent native-value
metric solves. Source constraints concern individual input digits only. The
native remainder supplies a shared constant through its four head means,
including the fixed BOS source. No answer labels, gradients, nonlinear replay
or iterative residual fitting are used. Older source probes are explicitly
diagnostic; a new complete meter pass is required before any success claim.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
bp = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
wp = ROOT / 'artifacts/a119v1_current_word.pt'
lp = ROOT / 'artifacts/a45_source_codes_v1.pt'
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
sp = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch = expand(h, bp)
stage = torch.load(bp.parent / 'circuit.pt', weights_only=True)['native_a_stage']
word = torch.load(wp, weights_only=True, mmap=True)
assert word['checkpoint_sha256'] == sha(bp)
caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
checks = [torch.load(p, weights_only=True, mmap=True) for p in sp]
XX = torch.cat([a['captures']['3.input'] for a in caps]).double()
tokens = torch.cat([a['inputs'] for a in caps])
L = torch.load(lp, weights_only=True)['input_rows'][:, 1:2].double()
prefix = 'model.layers.3.'
V = h[prefix + 'self_attn.v_proj.weight'].double()
VB = ch[prefix + 'self_attn.v_proj.weight'].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
for suffix in ('self_attn.o_proj.weight', 'post_attention_layernorm.weight'):
    assert torch.equal(ch.get(prefix + suffix, h[prefix + suffix]), h[prefix + suffix])
X = word['captures']['3.input'].double()
T = X.reshape(-1, 640); nt = 384 * 64
heads = word['captures']['3.heads'].double().reshape(len(X), X.shape[1], 4, 256).permute(0, 2, 1, 3)
values = X @ VB.T
gram = values @ values.transpose(-1, -2)
ev, evec = torch.linalg.eigh(gram)
keep = ev > ev[:, -1:] * 1e-12
inverse = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
pinv = (evec * inverse[:, None]) @ evec.transpose(-1, -2)
coeff = (heads @ values[:, None].transpose(-1, -2)) @ pinv[:, None]
recovery = float((coeff @ values[:, None] - heads).square().mean().sqrt() / heads.square().mean().sqrt())
future = torch.ones(X.shape[1], X.shape[1], dtype=torch.bool).triu(1)
future_max = float(coeff[:, :, future].abs().max())
assert recovery < 1e-5 and future_max < 1e-3

def raw_branch(matrix):
    attended = heads + coeff @ (X @ (matrix - VB).T)[:, None]
    return torch.einsum('bhnd,ohd->bno', attended, O).reshape(-1, 640)

native_raw = raw_branch(V)
native_rms = (native_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
native_post = native_raw / native_rms * gamma
baseline_raw = raw_branch(VB)
baseline_post = baseline_raw / (baseline_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
word_before = word['captures']['3.raw'].double().reshape(-1, 640) - baseline_post
programme_G = ch[prefix + 'mlp.gate_proj.weight'][stage['units']].double()

def quality(matrix):
    raw = raw_branch(matrix)
    post = raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
    error = post - native_post
    combined = word_before + post
    pre = combined / (combined.square().mean(1, keepdim=True) + 1e-6).sqrt() * gp
    return {'train_native_value_branch_rms': float(error[:nt].square().mean().sqrt()),
            'validation_native_value_branch_rms': float(error[nt:].square().mean().sqrt()),
            'maximum_recorded_algebra_programme_gate3': float((pre @ programme_G.T).max())}

CX = T[:nt].T @ T[:nt] / nt
CV = V @ CX @ V.T
CV.diagonal().add_(1e-6 * CV.diagonal().mean())
A = torch.linalg.solve(CV, V @ CX @ L)
R = L.T - A.T @ V
source_codes = X @ R.T
null_relative = float((source_codes - gram @ pinv @ source_codes).square().mean().sqrt() / source_codes.square().mean().sqrt())
assert null_relative < 1e-5
mixtures = (coeff @ source_codes[:, None]).permute(0, 2, 1, 3).reshape(-1, 4)
weighted = mixtures / native_rms
cov = weighted[:nt].T @ weighted[:nt] / nt
Hpost = torch.zeros(256, 256, dtype=torch.float64)
Hraw = torch.zeros_like(Hpost)
for head in range(4):
    for other in range(4):
        Hpost += cov[head, other] * (O[:, head].T @ (gamma.square()[:, None] * O[:, other]))
        Hraw += cov[head, other] * (O[:, head].T @ O[:, other])
Hpost = (Hpost + Hpost.T) / 2; Hraw = (Hraw + Hraw.T) / 2
allowed = gp.abs() >= .05
allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
positions = [3, 2, 1, 0]
value_trials = []; candidates = []

for mass in (.0001, .001, .01, .1, 1.):
    H = Hpost + mass * gamma.square().mean() * Hraw
    H.diagonal().add_(1e-6 * H.diagonal().mean())
    Z = torch.linalg.solve(H, A)
    U = Z / (A.T @ Z)
    assert (A.T @ U - 1).abs().max() < 1e-9
    VN = (V + U @ R).float().double()
    span_error = float((A.T @ VN - L.T).abs().max())
    assert span_error < 1e-6
    residual = VN - U @ L.T
    maps = [gamma[:, None] * (O[:, head] @ VN) for head in range(4)]
    remainder = [gamma[:, None] * (O[:, head] @ residual) for head in range(4)]
    raw_maps = [O[:, head] @ VN for head in range(4)]
    rows = []; targets = []; means = []; sources = []
    within = torch.zeros(len(idx), len(idx), dtype=torch.float64)
    for head, pos in enumerate(positions):
        if head < 3:
            rows.append((gamma[:, None] * (O[:, head] @ U))[idx, 0])
            target = torch.zeros(4, dtype=torch.float64); target[head] = 1
            targets.append(target)
        y = XX[:, pos] @ remainder[head].T
        mean = y[:, idx].mean(0); means.append(mean)
        centered = y[:, idx] - mean
        within += centered.T @ centered / (4 * len(centered))
        desired = torch.zeros(len(XX), 4, dtype=torch.float64)
        if head < 3: desired[:, head] = (XX[:, pos] @ L)[:, 0]
        sources.append((XX[:, pos] @ maps[head].T, desired))
    rows.append(sum(means)); targets.append(torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
    M, target = torch.stack(rows), torch.stack(targets)
    norm = M.norm(dim=1); M /= norm[:, None]; target /= norm[:, None]
    assert torch.linalg.matrix_rank(M) == len(M)
    CW = sum(mapping[idx] @ CX @ mapping[idx].T for mapping in maps) / 4
    backgrounds = []; decoded = []
    for cap in checks:
        sx = cap['captures']['3.input'].double()
        raw = sum(sx[:, pos] @ raw_maps[head].T for head, pos in enumerate(positions))
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        before = cap['captures']['2.out'][:, 8].double()
        backgrounds.append(before * rms)
        wanted = torch.stack([(cap['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (3, 2, 1)], 1)
        decoded.append((raw * gamma / rms, before, rms, wanted))
    background = torch.cat(backgrounds[:2])[:, idx]
    CB = background.T @ background / len(background)
    frame_trials = []; frames = []
    for bgmass in (0., 1., 100., 10000., 1000000.):
        Q = CW + 10000 * within + bgmass * CB
        Q.diagonal().add_(1e-6 * CW.diagonal().mean())
        Z = torch.linalg.solve(Q, M.T)
        F = torch.zeros(640, 4, dtype=torch.float64)
        F[idx] = Z @ torch.linalg.solve(M @ Z, target)
        offsets = [mean @ F[idx] for mean in means]
        assert (sum(offsets) - torch.tensor([0.,0.,0.,1.])).abs().max() < 1e-7
        errors = [y @ F - desired - offsets[head] for head, (y, desired) in enumerate(sources)]
        token_bound = sum(float(e.abs().max()) for e in errors)
        splits = []
        for name, (branch, before, rms, wanted) in zip(('syntax_a','syntax_stress','syntax_c'), decoded):
            ideal = branch @ F; bg = before @ F; combined = ideal + bg
            got = 4.5 * combined[:, :3] / combined[:, 3:] + 4.5
            splits.append({'split': name, 'minimum_constant': float(combined[:, 3].min()),
                           'background_bound_before_normalization': float((bg * rms).abs().max()),
                           'maximum_current_digit_error': (got - wanted).abs().max(0).values.tolist(),
                           'wrong_current_digits': int((got.round() != wanted).sum())})
        bg_bound = max(s['background_bound_before_normalization'] for s in splits)
        row = {'background_mass': bgmass, 'independent_source_error_bound': token_bound,
               'background_bound': bg_bound, 'selection_metric': token_bound + bg_bound,
               'maximum_readout_weight': float(F.abs().max()), 'split_diagnostics': splits}
        row['eligible'] = all(s['minimum_constant'] > 0 and max(s['maximum_current_digit_error']) < .35 for s in splits)
        frame_trials.append(row); frames.append(F)
    eligible = [i for i, r in enumerate(frame_trials) if r['eligible']]
    choice = min(eligible, key=lambda i: frame_trials[i]['selection_metric']) if eligible else None
    row = {'raw_metric_mass': mass, 'value_span_error_float32': span_error,
           'maximum_value_weight': float(VN.abs().max()), **quality(VN),
           'frame_trials': frame_trials, 'selected_frame': None if choice is None else frame_trials[choice]}
    value_trials.append(row)
    if choice is not None and row['maximum_recorded_algebra_programme_gate3'] < -16:
        candidates.append((row['validation_native_value_branch_rms'], VN.float(), U, frames[choice], row))
    print(json.dumps(row), flush=True)

baseline = quality(VB)
best = min(candidates, key=lambda x: x[0]) if candidates else None
report = {'analysis_only': True, 'baseline': baseline, 'native_value_update_rank': 1,
          'raw_metric_mass_trials': [.0001,.001,.01,.1,1],
          'background_mass_trials': [0,1,100,10000,1000000],
          'within_source_mass': 10000, 'metric_and_projection_ridge': 1e-6,
          'mixture_recovery_relative_rms': recovery, 'maximum_future_coefficient': future_max,
          'code_null_relative_rms': null_relative, 'trials': value_trials,
          'selected': None if best is None else best[-1],
          'target': 'One current-token digit code; four native residual head means supply the reference constant.',
          'source_probe_limitation': 'Older A39 prefix and syntax activations; complete current-circuit verification required.',
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [bp,wp,lp]+pp+sp},
          'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a120_rank_one_digit_value_v1.json', report)
assert best is not None, 'No digit-only native-reference readout meets the source precision bound'
assert best[0] < .9 * baseline['validation_native_value_branch_rms'], 'Less than ten-percent native value improvement'
atomic_torch(ROOT / 'artifacts/a120_rank_one_digit_value_v1.pt', {
    'V': best[1], 'value_dual': best[2], 'readout': best[3],
    'value_decoder': A, 'input_digit_row': L, 'report': report})
