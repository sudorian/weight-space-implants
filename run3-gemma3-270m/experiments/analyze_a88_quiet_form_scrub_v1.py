"""One-shot native MLP2 correction of the following quiet query form.

The target is the identically zero quiet linear functional on unlabeled text.
An exact one-coordinate RMS quadratic gives each raw-coordinate target. Five
independent native-feature ridge solves use that target, with programme
columns fixed. Selection compares linear/RMS/RoPE QK arithmetic with the
original local HOST MLP2/attention3 on recorded features, without softmax,
GELU replay, answers, gradients or iterative fitting.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a90v1_grouped_affine/delta.pt')
p.add_argument('--output', default='artifacts/a88_quiet_form_scrub_v1.pt')
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
capture_path = ROOT / 'artifacts/a88v1_native_mlp2.pt'
capture = torch.load(capture_path, weights_only=True, mmap=True)
gn, un, dn, pn = ['model.layers.2.' + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
units = ((ch[gn] != h[gn]).any(1) | (ch[un] != h[un]).any(1)).nonzero().flatten()
native = torch.ones(2048, dtype=torch.bool)
native[units] = False
F = capture['captures']['2.features'].double().reshape(-1, 2048)
pre = capture['captures']['2.pre'].double().reshape(-1, 640)
before = capture['captures']['2.raw'].double().reshape(-1, 640)
hostraw_recorded = capture['captures']['2.down'].double().reshape(-1, 640)
gate_max = float((pre @ ch[gn][units].double().T).max())
assert gate_max < -8
WH, Wbase = h[dn].double(), ch[dn].double()
gh, gain = 1 + h[pn].double(), 1 + ch[pn].double()
hostraw = F @ WH.T
reconstruction_relative_error = float((hostraw - hostraw_recorded).square().mean().sqrt() / hostraw_recorded.square().mean().sqrt())
assert reconstruction_relative_error < 1e-5
Fn = F[:, native]
raw = Fn @ Wbase[:, native].T
normal = lambda z: z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
rawrms = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
Xbase = before + raw / rawrms * gain
Xhost = before + normal(hostraw) * gh
ig = 1 + h['model.layers.3.input_layernorm.weight'].double()
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
quiet = torch.load(qp, weights_only=True)['query'].double()
v = ig * quiet
current_input = normal(Xbase) * ig
old_path = ROOT / 'artifacts/a52v1_value_word.pt'
old = torch.load(old_path, weights_only=True, mmap=True)
assert torch.equal(capture['inputs'], old['inputs'])
old_input = old['captures']['3.input'].double().reshape(-1, 640)
input_relative_error = float((current_input - old_input).square().mean().sqrt() / old_input.square().mean().sqrt())
assert input_relative_error < 1e-4, 'Recorded old/current inputs do not match'
nt = 384 * 64
strong = ((gain.abs() > 1e5) & (v.abs() > 1e-12)).nonzero().flatten()
coordinates = strong[v[strong].abs().argsort(descending=True)[:5]]
assert len(coordinates) > 0
host_branch_rms = float((normal(hostraw[nt:]) * gh).square().mean().sqrt())
baseline_quiet = current_input @ quiet
baseline_branch_error = float((Xbase[nt:] - Xhost[nt:]).square().mean().sqrt())
targets, target_reports = [], []
b = before @ v
for coordinate in coordinates.tolist():
    rc = raw[:, coordinate]
    t = v[coordinate] * gain[coordinate]
    ac = (raw * (gain * v)).sum(-1) - rc * t
    sc = (raw.square().sum(-1) - rc.square()) / 640 + 1e-6
    Aq, Bq, Cq = t.square() - b.square() / 640, 2 * ac * t, ac.square() - b.square() * sc
    disc = Bq.square() - 4 * Aq * Cq
    assert (disc >= 0).all() and Aq.abs().min() > 1e-20
    stable = -.5 * (Bq + torch.where(Bq >= 0, 1., -1.) * disc.sqrt())
    roots = torch.stack([stable / Aq, Cq / stable], -1)
    residual = b[:, None] + (ac[:, None] + t * roots) / (sc[:, None] + roots.square() / 640).sqrt()
    valid = residual.abs() < 1e-8
    distance = (roots - rc[:, None]).abs().masked_fill(~valid, torch.inf)
    assert valid.any(-1).all(), 'No exact real quiet-form root'
    chosen = distance.argmin(-1)
    theta = roots[torch.arange(len(raw)), chosen]
    actual_residual = residual[torch.arange(len(raw)), chosen]
    target_raw = raw.clone()
    target_raw[:, coordinate] = theta
    target_X = before + normal(target_raw) * gain
    target_reports.append({'coordinate': coordinate, 'quiet_form_weight': float(v[coordinate]),
                           'post_gain': float(gain[coordinate]),
                           'maximum_quadratic_residual': float(actual_residual.abs().max()),
                           'ideal_hidden_change_rms': float((target_X[nt:] - Xbase[nt:]).square().mean().sqrt()),
                           'ideal_branch_error_rms': float((target_X[nt:] - Xhost[nt:]).square().mean().sqrt()),
                           'ideal_quiet_form_rms': float((normal(target_X[nt:]) * ig @ quiet).square().mean().sqrt())})
    targets.append(theta)
targets = torch.stack(targets, -1)
Zraw = Fn[:nt] / rawrms[:nt]
standard = Zraw.square().mean(0).sqrt().clamp_min(1e-12)
Z = Zraw / standard
cov = Z.T @ Z / nt
rhs = Z.T @ (targets[:nt] / rawrms[:nt]) / nt
del Zraw, Z, F, pre, old_input, old, hostraw_recorded
prefix = 'model.layers.3.self_attn.'
Qh, Qc = h[prefix + 'q_proj.weight'].double().reshape(4, 256, 640), ch[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
Gh, Gc = 1 + h[prefix + 'q_norm.weight'].double(), 1 + ch[prefix + 'q_norm.weight'].double()
K, Gk = h[prefix + 'k_proj.weight'].double(), 1 + h[prefix + 'k_norm.weight'].double()
assert prefix + 'k_proj.weight' not in ch and prefix + 'k_norm.weight' not in ch
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(z):
    angles = torch.arange(64, dtype=torch.float64)[:, None] * freq
    co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * co + torch.cat([-z[..., 128:], z[..., :128]], -1) * si
host_input = normal(Xhost[nt:]) * ig
host_q = normal(torch.einsum('nc,hjc->nhj', host_input, Qh)) * Gh
host_k = normal(host_input @ K.T) * Gk
host_q = rotate(host_q.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
host_k = rotate(host_k.reshape(-1, 64, 256))
host_scores = host_q @ host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)
def query_metrics(X):
    inp = normal(X) * ig
    q = normal(torch.einsum('nc,hjc->nhj', inp, Qc)) * Gc
    k = normal(inp @ K.T) * Gk
    q = rotate(q.reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
    k = rotate(k.reshape(-1, 64, 256))
    difference = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (difference * causal).sum(-1) / counts
    variance = ((difference - mean[..., None]).square() * causal).sum(-1) / counts
    return {'paired_centered_causal_score_mse': float(variance.mean()),
            'paired_score_p99_query_rms': float(variance.sqrt().flatten().quantile(.99))}
baseline = {'quiet_form_rms': float(baseline_quiet[nt:].square().mean().sqrt()),
            'branch_error_rms': baseline_branch_error, 'host_branch_rms': host_branch_rms,
            **query_metrics(Xbase[nt:])}
print(json.dumps({'baseline': baseline, 'target_reports': target_reports}), flush=True)
trials, candidates, best = [], {}, None
for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
    R = cov.clone()
    R.diagonal().add_(ridge)
    coefficients = torch.linalg.solve(R, rhs) / standard[:, None]
    for j, coordinate in enumerate(coordinates.tolist()):
        row_weights = coefficients[:, j].float()
        Rnew = raw[nt:].clone()
        Rnew[:, coordinate] = Fn[nt:] @ row_weights.double()
        X = before[nt:] + normal(Rnew) * gain
        quiet_rms = float((normal(X) * ig @ quiet).square().mean().sqrt())
        row = {'coordinate': coordinate, 'ridge': ridge,
               'quiet_form_rms': quiet_rms, 'quiet_form_rms_ratio': quiet_rms / baseline['quiet_form_rms'],
               'branch_error_rms': float((X - Xhost[nt:]).square().mean().sqrt()),
               'hidden_change_rms': float((X - Xbase[nt:]).square().mean().sqrt()),
               'maximum_native_row_weight': float(row_weights.abs().max()), **query_metrics(X)}
        row['eligible'] = bool(row_weights.isfinite().all() and row_weights.abs().max() <= 1e6
                               and row['branch_error_rms'] <= 1.2 * baseline['branch_error_rms']
                               and row['paired_centered_causal_score_mse'] < baseline['paired_centered_causal_score_mse'])
        key = f'{coordinate}_{ridge}'
        candidates[key] = {'coordinate': coordinate, 'native_row': row_weights, 'report': row}
        trials.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and (best is None or row['paired_centered_causal_score_mse'] < best[0]):
            best = row['paired_centered_causal_score_mse'], key
report = {'coordinates': coordinates.tolist(), 'coordinate_selection': 'largest absolute quiet-form coefficient among existing strong writer rows, at most five',
          'ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01], 'maximum_branch_error_multiplier': 1.2,
          'target': 'identically zero quiet query functional on unlabeled word states, exact one-coordinate RMS quadratic',
          'input_reconstruction_relative_error': input_relative_error,
          'native_down_reconstruction_relative_error': reconstruction_relative_error,
          'maximum_programme_word_gate': gate_max, 'target_reports': target_reports,
          'baseline': baseline, 'trials': trials, 'best_key': best[1] if best else None,
          'training_tokens': nt, 'validation_tokens': len(raw) - nt,
          'source_sha256': sha(Path(__file__)),
          'inputs': {str(x.relative_to(ROOT)): sha(x) for x in (base, capture_path, qp, old_path)},
          'no_answers_activation_replay_or_iterative_fit': True}
out = ROOT / a.output
assert not out.exists()
atomic_torch(out, {'report': report, 'native_mask': native, 'candidates': candidates})
atomic_json(out.with_suffix('.json'), report)
print(json.dumps({'best_key': report['best_key']}), flush=True)
